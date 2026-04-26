import { Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/Toaster'
import Layout from '@/components/Layout'
import ProjectLayout from '@/components/ProjectLayout'
import ProjectsPage from '@/pages/ProjectsPage'
import ProjectPage from '@/pages/ProjectPage'
import AnnotatePage from '@/pages/AnnotatePage'
import VideoAnnotatePage from '@/pages/VideoAnnotatePage'
import TrainingPage from '@/pages/TrainingPage'
import InferencePage from '@/pages/InferencePage'
import InferenceFrameSelectPage from '@/pages/InferenceFrameSelectPage'
import ZCalibrationPage from '@/pages/ZCalibrationPage'
import TrackingComparePage from '@/pages/TrackingComparePage'

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/projects" replace />} />
          <Route path="projects" element={<ProjectsPage />} />
          <Route path="projects/:projectName" element={<ProjectLayout />}>
            <Route index element={<ProjectPage />} />
            <Route path="train" element={<TrainingPage />} />
            <Route path="inference" element={<InferencePage />} />
          </Route>
          <Route path="projects/:projectName/annotate" element={<AnnotatePage />} />
          <Route path="projects/:projectName/annotate/video/:videoId" element={<VideoAnnotatePage />} />
          <Route path="projects/:projectName/inference/:runName/:videoId/:inferenceId/frames" element={<InferenceFrameSelectPage />} />
          <Route path="projects/:projectName/inference/:runName/:videoId/:inferenceId/z-calibration" element={<ZCalibrationPage />} />
          <Route path="projects/:projectName/inference/:runName/:videoId/:inferenceId/tracking-compare" element={<TrackingComparePage />} />
        </Route>
      </Routes>
      <Toaster />
    </>
  )
}

export default App

