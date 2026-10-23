# PeopleWelcome

PeopleWelcome is a locally hosted, persona-based AI chat and image platform. It uses:

- ZITADEL OpenID Connect for sign-in
- PostgreSQL for users, characters, personas, conversations, and subscription state
- Stripe Billing for the $20 USD/month PeopleWelcome Premium subscription
- OpenAI and the Premium Unstoppable model hosted on Modal
- a local NVIDIA image service using Stable Diffusion 1.5, OpenPose ControlNet, and IP-Adapter identity guidance

## Local setup

1. Copy `.env.example` to `.env` and keep `.env` out of source control.
2. Configure the ZITADEL and Stripe resources described below.
3. Add the resulting IDs and secrets to `.env`.
4. Run `docker compose up --build`.
5. Open http://localhost:13000. The API is at http://localhost:18000.

PostgreSQL data persists in the `peoplewelcome-postgres` Docker volume. Uploaded and generated images persist in `peoplewelcome-data`. On the first PostgreSQL startup, the scheduler imports characters and messages from the old `/data/peoplewelcome.db` if one exists. Old simulated Premium flags are intentionally not imported.

After first ZITADEL sign-in, open **Settings** and select **Import local characters** to deliberately move the old `demo-user` and `local-user` records into your account. Keep `LEGACY_IMPORT_ENABLED=true` only while using the app locally, then disable it.

## ZITADEL configuration

Create a ZITADEL project with two applications:

1. A **User Agent / SPA** application using Authorization Code with PKCE. Add:
   - redirect URI: `http://localhost:13000/auth/callback`
   - post-logout redirect URI: `http://localhost:13000/login`
   - development mode for local HTTP only
2. An **API** application with Basic authentication for token introspection. Save its client ID and client secret.

Set these values in `.env`:

```env
ZITADEL_ISSUER=https://your-instance.zitadel.cloud
ZITADEL_PROJECT_ID=your-project-id
ZITADEL_FRONTEND_CLIENT_ID=your-spa-client-id
ZITADEL_INTROSPECTION_CLIENT_ID=your-api-client-id
ZITADEL_INTROSPECTION_CLIENT_SECRET=your-api-client-secret
ZITADEL_AUDIENCE=your-project-id
```

The browser requests `openid profile email offline_access` plus the ZITADEL project audience scope. The API introspects every new access token and derives the user ID from the validated `sub` claim; it never trusts a browser-supplied user ID.

There is no built-in demo password after this change. Create a test user in ZITADEL and use that user's password on the ZITADEL-hosted sign-in page.

## Stripe Premium configuration

In Stripe, create a recurring Price with all of these properties:

- USD $20.00
- recurring monthly
- active

Copy its `price_...` ID into `STRIPE_PREMIUM_PRICE_ID`, and add your server secret key to `STRIPE_SECRET_KEY`. The API validates the Price before starting Checkout, so a different amount or interval is rejected.

Create a webhook endpoint for:

`http://localhost:18000/api/v1/billing/webhook`

Subscribe it to these events:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `invoice.paid`
- `invoice.payment_failed`

Put the endpoint's `whsec_...` signing secret in `STRIPE_WEBHOOK_SECRET`. For local Stripe CLI testing:

```text
stripe listen --forward-to localhost:18000/api/v1/billing/webhook
```

Use the `whsec_...` value printed by that command while the listener is running. Premium becomes active only after a signed webhook reports an `active` or `trialing` subscription for the configured Price. Checkout success by itself does not grant access.

## Character views and scenes

Uploading an avatar during character creation starts five separate generations:

1. full-body front
2. full-body side
3. full-body back
4. face close-up front
5. face close-up side

The uploaded image is used for identity through IP-Adapter. A different fixed OpenPose control image is used for every requested camera view, so the service synthesizes a new view rather than lightly editing the uploaded pixels. Appearance details in the persona are included in every image prompt.

The Diffusers pipeline is deliberately created without a safety checker. It does not replace normal images with black output and does not implement an NSFW-refusal path.

Docker Desktop must have NVIDIA GPU access enabled. The first generation downloads the base model, OpenPose ControlNet, IP-Adapter, and image encoder into the shared Hugging Face cache, so it is much slower than later requests. Useful health checks:

```text
http://localhost:18000/health
docker compose exec image-generator python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Model API keys and the Stripe/ZITADEL server secrets are passed only to the scheduler container. They are never embedded in browser code. The ZITADEL SPA client ID and project ID are public identifiers and are embedded into the frontend at build time.
