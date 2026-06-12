import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Sidebar } from './components/layout/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { ExperimentPage } from './pages/ExperimentPage';
import { CreateExperiment } from './pages/CreateExperiment';
import { PowerAnalysis } from './pages/PowerAnalysis';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex min-h-screen bg-slate-50">
          <Sidebar />
          <main className="flex-1 ml-60 p-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/experiments/new" element={<CreateExperiment />} />
              <Route path="/experiments/:id" element={<ExperimentPage />} />
              <Route path="/power" element={<PowerAnalysis />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
