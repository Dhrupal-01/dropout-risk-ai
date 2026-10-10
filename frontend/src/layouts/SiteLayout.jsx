import React, { Suspense } from 'react';
import { Link, Outlet } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import SkipLink from '../components/SkipLink';
import PageLoading from '../components/PageLoading';
import { REPO_URL, isDemoMode } from '../content/site';

// Public site: top nav + footer. Section anchor links (#problem, #how-it-works, #evidence,
// #responsible-ai) are added to the nav as each landing section is built.
const SiteLayout = () => (
  <div className="min-h-screen flex flex-col bg-paper text-graphite">
    <SkipLink />

    <header className="border-b border-rule">
      <div className="max-w-page mx-auto px-4 sm:px-6 py-4 flex items-center justify-between gap-3">
        <Link to="/" className="font-display font-medium text-20 sm:text-24 tracking-display text-graphite">
          DropoutGuard
        </Link>
        <nav aria-label="Site" className="flex items-center gap-2 sm:gap-3">
          {/* Plain anchors: from other pages they load the landing page at the section */}
          <a href="/#problem" className="hidden sm:inline-block px-2 py-1 text-15 text-graphite hover:text-ink">
            The problem
          </a>
          <ThemeToggle />
          <Link to="/app/overview" className="btn btn-primary">
            Open dashboard
          </Link>
        </nav>
      </div>
    </header>

    <main id="main" tabIndex={-1} className="flex-1 focus:outline-none">
      <Suspense fallback={<PageLoading />}>
        <Outlet />
      </Suspense>
    </main>

    <footer className="border-t border-rule">
      <div className="max-w-page mx-auto px-4 sm:px-6 py-10 text-15 text-slate space-y-2">
        <p className="font-medium text-graphite">DropoutGuard</p>
        <p>Smart India Hackathon 2026, SDG 4: Quality Education</p>
        <p>
          <a href={REPO_URL} className="text-ink underline underline-offset-4 hover:no-underline">
            Source code on GitHub
          </a>
        </p>
        {isDemoMode && <p>Demo data is simulated.</p>}
      </div>
    </footer>
  </div>
);

export default SiteLayout;
