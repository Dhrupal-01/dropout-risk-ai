import React, { Suspense } from 'react';
import { Link, NavLink, Outlet } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { LayoutDashboard, Users, Upload, ArrowLeft } from 'lucide-react';
import AdminTokenField from '../components/AdminTokenField';
import HealthBadge from '../components/HealthBadge';
import ThemeToggle from '../components/ThemeToggle';
import SkipLink from '../components/SkipLink';
import PageLoading from '../components/PageLoading';
import { isDemoMode } from '../content/site';

// Lives in the app chunk so public pages never load TanStack Query or the API client.
// Module scope keeps the cache across visits to the site and back.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 60000, // 60s cache validity
    },
  },
});

const NAV_ITEMS = [
  { to: '/app/overview', label: 'Overview', icon: LayoutDashboard },
  { to: '/app/students', label: 'Students', icon: Users },
  { to: '/app/import', label: 'Import', icon: Upload },
];

const navItemClass = ({ isActive }) =>
  `flex items-center gap-2 px-3 py-2 rounded-control text-15 whitespace-nowrap transition-colors ${
    isActive ? 'bg-ink-wash text-ink font-medium' : 'text-graphite hover:bg-ink-wash'
  }`;

const AppLayout = () => (
  <QueryClientProvider client={queryClient}>
    <div className="min-h-screen bg-paper text-graphite md:flex">
      <SkipLink />

      {/* Left rail on desktop; a horizontal bar on small screens */}
      <aside className="border-b md:border-b-0 md:border-r border-rule md:w-56 md:shrink-0">
        <div className="md:sticky md:top-0 md:max-h-screen md:overflow-y-auto">
          <div className="px-4 pt-4 md:pb-4 flex items-center justify-between gap-4">
            <Link to="/app/overview" className="font-display font-medium text-24 tracking-display text-graphite">
              DropoutGuard
            </Link>
            {/* Small screens: next to the brand, so the nav row never overflows */}
            <Link to="/" className="md:hidden flex items-center gap-1.5 text-15 text-slate hover:text-graphite">
              <ArrowLeft className="w-4 h-4 shrink-0" aria-hidden="true" />
              Back to site
            </Link>
          </div>
          <nav aria-label="App" className="px-2 pb-2 md:pb-4">
            <ul className="flex md:flex-col gap-1">
              {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
                <li key={to}>
                  <NavLink to={to} className={navItemClass}>
                    <Icon className="w-4 h-4 shrink-0" aria-hidden="true" />
                    {label}
                  </NavLink>
                </li>
              ))}
              <li role="separator" aria-hidden="true" className="hidden md:block border-t border-rule mx-3 my-2" />
              <li className="hidden md:block">
                <Link to="/" className="flex items-center gap-2 px-3 py-2 rounded-control text-15 whitespace-nowrap text-slate hover:bg-ink-wash hover:text-graphite transition-colors">
                  <ArrowLeft className="w-4 h-4 shrink-0" aria-hidden="true" />
                  Back to site
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        {isDemoMode && (
          <div className="bg-amber-500/10 dark:bg-amber-500/20 border-b border-amber-500/30 text-amber-800 dark:text-amber-200 text-xs py-1.5 px-4 text-center font-medium select-none">
            Demo environment — all student records are simulated.
          </div>
        )}

        <header className="sticky top-0 z-40 bg-paper border-b border-rule px-4 sm:px-6 py-3 flex flex-wrap items-center justify-end gap-x-4 gap-y-2">
          <AdminTokenField />
          <div className="flex items-center gap-2 text-13 text-slate">
            <span className="hidden sm:inline">Backend:</span>
            <HealthBadge />
          </div>
          <ThemeToggle />
        </header>

        <main id="main" tabIndex={-1} className="flex-1 tabular-nums focus:outline-none">
          <Suspense fallback={<PageLoading />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  </QueryClientProvider>
);

export default AppLayout;
