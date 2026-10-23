import React, { useCallback, useContext, useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Settings() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { currentUser, authorizedFetch, signOut } = useContext(AuthContext);
  const [billing, setBilling] = useState(null);
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(false);
  const [notice, setNotice] = useState('');
  const checkoutResult = searchParams.get('checkout');

  const loadStatus = useCallback(async () => {
    const response = await authorizedFetch(`${API_BASE}/api/v1/billing/status`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load subscription status.');
    setBilling(data);
    return data;
  }, [authorizedFetch]);

  useEffect(() => {
    loadStatus().catch((error) => setNotice(error.message));
  }, [loadStatus]);

  useEffect(() => {
    if (checkoutResult === 'cancelled') {
      setNotice('Checkout was cancelled. No subscription change was made.');
      return undefined;
    }
    if (checkoutResult !== 'success') return undefined;
    setNotice('Payment received. Waiting for Stripe to confirm your subscription...');
    let attempts = 0;
    const poll = async () => {
      attempts += 1;
      try {
        const status = await loadStatus();
        if (status.premium) {
          setNotice('Premium is active. Unstoppable is now available when you create an AI character.');
          setSearchParams({}, { replace: true });
          window.clearInterval(timer);
        } else if (attempts >= 15) {
          setNotice('Stripe is still confirming the subscription. Refresh this page in a moment.');
          window.clearInterval(timer);
        }
      } catch (error) {
        setNotice(error.message);
        window.clearInterval(timer);
      }
    };
    const timer = window.setInterval(poll, 2000);
    poll();
    return () => window.clearInterval(timer);
  }, [checkoutResult, loadStatus, setSearchParams]);

  const openBillingUrl = async (endpoint) => {
    setBusy(true);
    setNotice('');
    try {
      const response = await authorizedFetch(`${API_BASE}${endpoint}`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Stripe is temporarily unavailable.');
      window.location.assign(data.url);
    } catch (error) {
      setNotice(error.message);
      setBusy(false);
    }
  };

  const importLegacyCharacters = async () => {
    setImporting(true);
    setNotice('');
    try {
      const response = await authorizedFetch(`${API_BASE}/api/v1/legacy/import`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not import local characters.');
      setNotice(`Imported ${data.imported.ais} character${data.imported.ais === 1 ? '' : 's'} and ${data.imported.messages} saved message${data.imported.messages === 1 ? '' : 's'}. Return to chat to see them.`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setImporting(false);
    }
  };

  const periodEnd = billing?.current_period_end
    ? new Date(billing.current_period_end).toLocaleDateString()
    : null;

  return (
    <main className="settings">
      <h1>Settings</h1>
      <section style={{ margin: '2rem 0', padding: '1.5rem', border: '1px solid #ddd', borderRadius: '12px', maxWidth: '680px' }}>
        <h2>PeopleWelcome Premium</h2>
        <p><strong>$20 USD per month</strong></p>
        <p>Premium unlocks the Unstoppable model hosted on Modal when creating and chatting with AI characters.</p>

        {!billing && <p>Loading subscription...</p>}
        {billing?.premium && (
          <>
            <p role="status">Premium is active{periodEnd ? ` through ${periodEnd}` : ''}.</p>
            {billing.cancel_at_period_end && <p>Your subscription is scheduled to end at the close of this billing period.</p>}
            <button type="button" disabled={busy} onClick={() => openBillingUrl('/api/v1/billing/portal')}>
              {busy ? 'Opening...' : 'Manage subscription'}
            </button>
          </>
        )}
        {billing && !billing.premium && (
          <>
            <p>Premium is not active.</p>
            <button type="button" disabled={busy} onClick={() => openBillingUrl('/api/v1/billing/checkout')}>
              {busy ? 'Opening secure checkout...' : 'Subscribe for $20/month'}
            </button>
            {billing.has_customer && (
              <button type="button" disabled={busy} onClick={() => openBillingUrl('/api/v1/billing/portal')} style={{ marginLeft: '0.75rem' }}>
                Manage billing
              </button>
            )}
          </>
        )}
        {notice && <p role="status">{notice}</p>}
      </section>

      <section style={{ margin: '2rem 0' }}>
        <h2>Account</h2>
        <p>{currentUser?.displayName}{currentUser?.email ? ` (${currentUser.email})` : ''}</p>
        <p>Your password and profile are managed by ZITADEL.</p>
        <button type="button" onClick={() => signOut()}>Log out</button>
      </section>

      <section style={{ margin: '2rem 0', padding: '1.5rem', border: '1px solid #ddd', borderRadius: '12px', maxWidth: '680px' }}>
        <h2>Local character import</h2>
        <p>Move characters created before ZITADEL sign-in into this account. This only imports the old local demo records.</p>
        <button type="button" disabled={importing} onClick={importLegacyCharacters}>
          {importing ? 'Importing...' : 'Import local characters'}
        </button>
      </section>

      <button type="button" onClick={() => navigate(-1)}>Back</button>
    </main>
  );
}

export default Settings;
