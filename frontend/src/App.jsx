import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation, useParams } from 'react-router-dom';
import SiteLayout from './layouts/SiteLayout';
import PageLoading from './components/PageLoading';
import RouteFocus from './components/RouteFocus';

// Every route is code-split; public pages never load the app layout, TanStack Query or the API client.
const AppLayout = lazy(() => import('./layouts/AppLayout'));
const Home = lazy(() => import('./pages/site/Home'));
const NotFound = lazy(() => import('./pages/NotFound'));
const Overview = lazy(() => import('./pages/app/Overview'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const StudentDetail = lazy(() => import('./pages/StudentDetail'));
const ImportQueue = lazy(() => import('./pages/ImportQueue'));

// Redirects keep the query string and hash (worklist filters live in the query string).
const RedirectTo = ({ to }) => {
  const { search, hash } = useLocation();
  return <Navigate replace to={{ pathname: to, search, hash }} />;
};

const LegacyStudentRedirect = () => {
  const { studentId } = useParams();
  return <RedirectTo to={`/app/students/${encodeURIComponent(studentId)}`} />;
};

function App() {
  return (
    <BrowserRouter>
      <RouteFocus />
      <Suspense fallback={<PageLoading />}>
        <Routes>
          <Route element={<SiteLayout />}>
            <Route index element={<Home />} />
            <Route path="*" element={<NotFound />} />
          </Route>

          <Route path="/app" element={<AppLayout />}>
            <Route index element={<RedirectTo to="/app/overview" />} />
            <Route path="overview" element={<Overview />} />
            <Route path="students" element={<Dashboard />} />
            <Route path="students/:studentId" element={<StudentDetail />} />
            <Route path="import" element={<ImportQueue />} />
            <Route path="worklist" element={<RedirectTo to="/app/students" />} />
          </Route>

          {/* URLs from before the redesign */}
          <Route path="/students/:studentId" element={<LegacyStudentRedirect />} />
          <Route path="/import" element={<RedirectTo to="/app/import" />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
