import React from 'react';
import { KeyRound } from 'lucide-react';
import { setAdminToken, useAdminToken } from '../api/adminToken';

// Admin token for writes: kept in memory only (src/api/adminToken.js), never stored
const AdminTokenField = () => {
  const adminToken = useAdminToken();

  return (
    <label className="flex w-full sm:w-auto items-center gap-1.5 text-13 font-medium text-slate">
      <KeyRound className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
      <span className="whitespace-nowrap">Admin token</span>
      <input
        type="password"
        autoComplete="off"
        spellCheck={false}
        value={adminToken}
        onChange={(e) => setAdminToken(e.target.value)}
        placeholder="needed for changes"
        title="Needed to score students, upload CSVs and log interventions. Kept in memory only; cleared on reload."
        className="min-w-0 flex-1 sm:flex-none sm:w-40 px-2 py-1 rounded-control border border-control bg-paper text-graphite text-13 font-normal placeholder:text-slate"
      />
    </label>
  );
};

export default AdminTokenField;
