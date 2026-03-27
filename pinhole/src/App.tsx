import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { DepthModel } from './pages/DepthModel';
import { PinholeModel } from './pages/PinholeModel';

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return null;
}

function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/depth-model" element={<DepthModel />} />
          <Route path="/pinhole-model" element={<PinholeModel />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
