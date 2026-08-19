import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import StudentDetail from './pages/StudentDetail';
import ImportQueue from './pages/ImportQueue';

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

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col bg-page text-primary transition-colors">
          {/* Global Header */}
          <Header />

          {/* Main Content Area */}
          <main className="flex-grow">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/students/:studentId" element={<StudentDetail />} />
              <Route path="/import" element={<ImportQueue />} />
              {/* Fallback path redirects to dashboard */}
              <Route path="*" element={<Dashboard />} />
            </Routes>
          </main>

          {/* Global Footer (Restrained / Professional) */}
          <footer className="py-6 border-t border-border bg-card text-center select-none text-[11px] text-muted">
            <div className="container mx-auto px-6">
              © {new Date().getFullYear()} DropoutGuard Systems · AI-Powered Academic Early-Warning &amp; Intervention · Smart India Hackathon
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
