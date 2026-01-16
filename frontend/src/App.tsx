import { Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/Toaster'
import Layout from '@/components/Layout'
import ProjectsPage from '@/pages/ProjectsPage'
import ProjectPage from '@/pages/ProjectPage'
import AnnotatePage from '@/pages/AnnotatePage'
import TrainingPage from '@/pages/TrainingPage'
import InferencePage from '@/pages/InferencePage'

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/projects" replace />} />
          <Route path="projects" element={<ProjectsPage />} />
          <Route path="projects/:projectName" element={<ProjectPage />} />
          <Route path="projects/:projectName/annotate" element={<AnnotatePage />} />
          <Route path="projects/:projectName/train" element={<TrainingPage />} />
          <Route path="projects/:projectName/inference" element={<InferencePage />} />
        </Route>
      </Routes>
      <Toaster />
    </>
  )
}

export default App

