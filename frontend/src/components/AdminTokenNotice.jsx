import React from 'react';
import { KeyRound } from 'lucide-react';
import { ADMIN_TOKEN_MISSING_MESSAGE, useAdminToken } from '../api/adminToken';

// Shown next to a write action while no admin token is entered.
const AdminTokenNotice = () => {
  const adminToken = useAdminToken();
  if (adminToken.trim()) return null;
  return (
    <p className="flex items-start gap-2 rounded-control border border-rule bg-ink-wash p-3 text-15 text-graphite">
      <KeyRound className="mt-0.5 w-4 h-4 shrink-0 text-ink" aria-hidden="true" />
      <span>{ADMIN_TOKEN_MISSING_MESSAGE}</span>
    </p>
  );
};

export default AdminTokenNotice;
