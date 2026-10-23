"""Stripe Premium checkout, portal, and webhook synchronization."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import stripe


class BillingConfigurationError(RuntimeError):
    pass


class BillingService:
    def __init__(self, store) -> None:
        self.store = store
        self.secret_key = os.getenv("STRIPE_SECRET_KEY", "")
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
        self.price_id = os.getenv("STRIPE_PREMIUM_PRICE_ID", "")
        self.app_base_url = os.getenv("APP_BASE_URL", "http://localhost:13000").rstrip("/")
        stripe.api_key = self.secret_key

    @property
    def configured(self) -> bool:
        return bool(self.secret_key and self.price_id)

    def require_configured(self) -> None:
        if not self.configured:
            raise BillingConfigurationError("Stripe Premium billing is not configured.")

    def validate_price(self) -> None:
        self.require_configured()
        price = stripe.Price.retrieve(self.price_id)
        recurring = price.get("recurring") or {}
        if not price.get("active") or price.get("currency") != "usd" or price.get("unit_amount") != 2000:
            raise BillingConfigurationError("STRIPE_PREMIUM_PRICE_ID must be an active USD $20.00 price.")
        if recurring.get("interval") != "month" or recurring.get("interval_count", 1) != 1:
            raise BillingConfigurationError("STRIPE_PREMIUM_PRICE_ID must recur once per month.")

    def create_checkout(self, user: dict) -> str:
        self.validate_price()
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(
                email=user.get("email") or None,
                name=user.get("username") or None,
                metadata={"zitadel_user_id": user["id"]},
            )
            customer_id = customer.id
            self.store.set_stripe_customer(user["id"], customer_id)
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            client_reference_id=user["id"],
            line_items=[{"price": self.price_id, "quantity": 1}],
            success_url=f"{self.app_base_url}/settings?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{self.app_base_url}/settings?checkout=cancelled",
            subscription_data={"metadata": {"zitadel_user_id": user["id"]}},
            metadata={"zitadel_user_id": user["id"]},
        )
        return session.url

    def create_portal(self, user: dict) -> str:
        self.require_configured()
        if not user.get("stripe_customer_id"):
            raise ValueError("No Stripe customer exists for this user.")
        session = stripe.billing_portal.Session.create(
            customer=user["stripe_customer_id"],
            return_url=f"{self.app_base_url}/settings",
        )
        return session.url

    @staticmethod
    def _timestamp(value) -> datetime | None:
        return datetime.fromtimestamp(value, tz=timezone.utc) if value else None

    @staticmethod
    def _period_end(subscription: dict) -> int | None:
        if subscription.get("current_period_end"):
            return subscription["current_period_end"]
        items = ((subscription.get("items") or {}).get("data") or [])
        return max((item.get("current_period_end") or 0 for item in items), default=0) or None

    def _matches_premium_price(self, subscription: dict) -> bool:
        items = ((subscription.get("items") or {}).get("data") or [])
        return any((item.get("price") or {}).get("id") == self.price_id for item in items)

    def sync_subscription(self, subscription: dict) -> dict | None:
        if not self._matches_premium_price(subscription):
            return None
        customer_id = subscription.get("customer")
        user = self.store.get_user_by_customer(customer_id) if customer_id else None
        metadata = subscription.get("metadata") or {}
        user_id = metadata.get("zitadel_user_id") or (user and user["id"])
        if not user_id:
            return None
        return self.store.update_subscription(
            user_id=user_id,
            customer_id=customer_id,
            subscription_id=subscription["id"],
            status=subscription.get("status", "inactive"),
            current_period_end=self._timestamp(self._period_end(subscription)),
            cancel_at_period_end=bool(subscription.get("cancel_at_period_end")),
        )

    def process_webhook(self, payload: bytes, signature: str | None) -> None:
        if not self.webhook_secret:
            raise BillingConfigurationError("STRIPE_WEBHOOK_SECRET is not configured.")
        event = stripe.Webhook.construct_event(payload, signature, self.webhook_secret)
        event_type = event["type"]
        obj = event["data"]["object"]
        if event_type == "checkout.session.completed" and obj.get("mode") == "subscription":
            user_id = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("zitadel_user_id")
            if user_id and obj.get("customer"):
                self.store.set_stripe_customer(user_id, obj["customer"])
            if obj.get("subscription"):
                self.sync_subscription(stripe.Subscription.retrieve(obj["subscription"]))
        elif event_type in {
            "customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"
        }:
            self.sync_subscription(obj)
        elif event_type in {"invoice.paid", "invoice.payment_failed"}:
            subscription_id = obj.get("subscription")
            if subscription_id:
                self.sync_subscription(stripe.Subscription.retrieve(subscription_id))
