import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import StudentDetail from './pages/StudentDetail';
import ImportQueue from './pages/ImportQueue';
import GeoAnalytics from './pages/GeoAnalytics';
import UniversalPredictor from './pages/UniversalPredictor';

// Initialize the query client with robust default retry and caching configurations
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 60000, // 60s cache validity
    },
  },
});

const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true';

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col bg-page text-primary transition-colors">
          {/* Demo Mode Persistent Banner */}
          {isDemoMode && (
            <div className="bg-amber-500/10 dark:bg-amber-500/20 border-b border-amber-500/30 text-amber-800 dark:text-amber-200 text-xs py-1.5 px-4 text-center font-medium select-none">
              Demo environment — all student records are simulated.
            </div>
          )}

          {/* Global Header */}
          <Header />

          {/* Main Content Area */}
          <main className="flex-grow">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/students/:studentId" element={<StudentDetail />} />
              <Route path="/import" element={<ImportQueue />} />
              <Route path="/geo-analytics" element={<GeoAnalytics />} />
              <Route path="/universal-predictor" element={<UniversalPredictor />} />
              {/* Fallback path redirects to dashboard */}
              <Route path="*" element={<Dashboard />} />
            </Routes>
          </main>

          {/* Global Footer (Restrained / Professional) */}
          <footer className="py-6 border-t border-border bg-card text-center select-none text-[11px] text-muted">
            <div className="container mx-auto px-6">
              © {new Date().getFullYear()} DropoutGuard Systems · AI-Powered Academic Early-Warning &amp; Intervention · DropoutGuard
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
