import React from 'react';
import { KeyRound } from 'lucide-react';
import { ADMIN_TOKEN_MISSING_MESSAGE, useAdminToken } from '../api/adminToken';

// Shown next to a write action while no admin token is entered.
const AdminTokenNotice = () => {
  const adminToken = useAdminToken();
  if (adminToken.trim()) return null;
  return (
    <p className="text-[11px] text-secondary bg-subtle/60 border border-border rounded p-2 flex items-start gap-1.5">
      <KeyRound className="w-3.5 h-3.5 shrink-0 mt-0.5 text-muted" aria-hidden="true" />
      <span>{ADMIN_TOKEN_MISSING_MESSAGE}</span>
    </p>
  );
};

export default AdminTokenNotice;
