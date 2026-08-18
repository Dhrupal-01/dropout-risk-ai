import React, { useState, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Upload, FileText, CheckCircle2, AlertOctagon, AlertTriangle, 
  ArrowLeft, RefreshCw, BarChart2, ShieldAlert
} from 'lucide-react';
import { uploadBatchCsv } from '../api/endpoints';

const ImportQueue = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Batch CSV prediction upload mutation
  const uploadMutation = useMutation({
    mutationFn: (uploadFile) => uploadBatchCsv(uploadFile, true),
    onSuccess: () => {
      // Refresh worklist queries
      queryClient.invalidateQueries(['queue']);
      queryClient.invalidateQueries(['kpi-high']);
      queryClient.invalidateQueries(['kpi-medium']);
      queryClient.invalidateQueries(['kpi-low']);
      queryClient.invalidateQueries(['kpi-total']);
    },
  });

  // Handle file select change
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  // Drag-and-drop events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
      } else {
        alert("Please upload a valid CSV file.");
      }
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    uploadMutation.mutate(file);
  };

  const triggerReset = () => {
    setFile(null);
    uploadMutation.reset();
  };

  const result = uploadMutation.data;
  const isError = uploadMutation.isError;
  const error = uploadMutation.error;

  return (
    <div className="container mx-auto px-6 py-8 max-w-2xl space-y-6">
      {/* Back to Worklist */}
      <div>
        <Link 
          to="/" 
          className="inline-flex items-center text-xs font-semibold text-secondary hover:text-primary transition-colors focus:ring-2 focus:ring-accent"
        >
          <ArrowLeft className="w-4 h-4 mr-1.5" />
          Back to Triage Worklist
        </Link>
      </div>

      <div className="bg-card border border-border p-6 rounded-lg shadow-sm space-y-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-primary">Bulk Student Prediction scoring</h1>
          <p className="text-xs text-secondary mt-0.5">
            Ingest and run batch predictive scoring on student cohorts by uploading a CSV data file.
          </p>
        </div>

        {!result && (
          /* CSV UPLOAD DRAG BOX */
          <form onSubmit={handleSubmit} className="space-y-4">
            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={onButtonClick}
              className={`border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center cursor-pointer transition-colors ${
                dragActive 
                  ? 'border-accent bg-accent-soft/20' 
                  : 'border-border bg-subtle/20 hover:border-accent/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
              
              <Upload className="w-8 h-8 text-muted mb-3" />
              
              {file ? (
                <div className="flex items-center space-x-2 bg-card border border-border p-2 rounded shadow-xs select-all">
                  <FileText className="w-4 h-4 text-accent shrink-0" />
                  <span className="text-xs font-semibold text-primary truncate max-w-xs">{file.name}</span>
                  <span className="text-[10px] text-muted">({(file.size / 1024).toFixed(1)} KB)</span>
                </div>
              ) : (
                <div className="text-center space-y-1 select-none">
                  <p className="text-xs font-bold text-primary">Drag and drop student CSV file here</p>
                  <p className="text-[10px] text-muted">or click to browse local files (CSV only)</p>
                </div>
              )}
            </div>

            <div className="text-[10px] text-muted leading-relaxed select-none">
              * The CSV file must contain a <code className="font-mono bg-subtle px-1 py-0.5 border border-border rounded text-primary">student_id</code> column and raw features matching the ML pipeline contract (e.g. attendance metrics, CGPA details, delay days). Engineered terms and labels are ignored.
            </div>

            <button
              type="submit"
              disabled={!file || uploadMutation.isLoading}
              className="w-full py-2.5 bg-accent hover:bg-accent-hover text-white text-xs font-semibold rounded-md shadow-xs transition-colors disabled:opacity-50 disabled:pointer-events-none inline-flex items-center justify-center"
            >
              {uploadMutation.isLoading && <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
              Execute Batch scoring Run
            </button>
          </form>
        )}

        {/* ERROR STATE */}
        {isError && (
          <div className="p-4 rounded-md border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/20 space-y-4">
            <div className="flex items-start">
              <ShieldAlert className="w-5 h-5 text-risk-high mr-3 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold text-red-800 dark:text-red-300">Scoring run failed</h4>
                <p className="text-xs text-red-700 dark:text-red-400 mt-1 select-text">
                  {error?.message || 'Schema parsing error. Please check feature columns and ranges.'}
                </p>
              </div>
            </div>
            <button
              onClick={triggerReset}
              className="w-full py-2 border border-red-300 dark:border-red-900 rounded bg-card hover:bg-hover text-xs font-semibold text-secondary transition-colors"
            >
              Try Uploading Again
            </button>
          </div>
        )}

        {/* SUCCESS SUMMARY RESULTS */}
        {result && (
          <div className="space-y-6 pt-2">
            <div className="flex items-center justify-center space-x-2 text-risk-low bg-green-50 dark:bg-green-950/10 border border-green-200 dark:border-green-900/30 p-3 rounded-lg select-none">
              <CheckCircle2 className="w-5 h-5 text-risk-low shrink-0" />
              <span className="text-xs font-semibold">Cohort Scored Successfully!</span>
            </div>

            {/* Metric results counts */}
            <div className="grid grid-cols-2 gap-4 border border-border p-4 rounded-lg bg-subtle/20">
              <div className="text-center">
                <span className="text-[10px] text-secondary uppercase font-semibold">Total Evaluated</span>
                <div className="text-2xl font-black text-primary font-mono mt-1">
                  {result.total_students_evaluated || result.results?.length || 0}
                </div>
              </div>
              <div className="text-center">
                <span className="text-[10px] text-secondary uppercase font-semibold">Model Version</span>
                <div className="text-xs font-mono text-secondary mt-2.5 truncate max-w-[180px] mx-auto bg-card border border-border p-1 rounded">
                  {result.model_version}
                </div>
              </div>
            </div>

            {/* Distribution metrics */}
            <div className="space-y-3">
              <span className="text-xs font-semibold text-secondary flex items-center gap-1.5 select-none">
                <BarChart2 className="w-4 h-4 text-accent" />
                Risk Tier Distribution
              </span>
              
              <div className="space-y-2">
                {/* High Risk */}
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center font-medium text-secondary">
                    <AlertOctagon className="w-3.5 h-3.5 mr-1.5 text-risk-high" /> High Risk
                  </span>
                  <span className="font-bold text-primary font-mono">{result.high_risk_count ?? 0}</span>
                </div>
                {/* Medium Risk */}
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center font-medium text-secondary">
                    <AlertTriangle className="w-3.5 h-3.5 mr-1.5 text-risk-medium" /> Medium Risk
                  </span>
                  <span className="font-bold text-primary font-mono">{result.medium_risk_count ?? 0}</span>
                </div>
                {/* Low Risk */}
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center font-medium text-secondary">
                    <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-risk-low" /> Low Risk
                  </span>
                  <span className="font-bold text-primary font-mono">{result.low_risk_count ?? 0}</span>
                </div>
              </div>
            </div>

            {/* CTA action buttons */}
            <div className="flex items-center space-x-3 pt-2">
              <button
                onClick={triggerReset}
                className="flex-1 py-2.5 border border-border rounded-md hover:bg-hover text-xs font-semibold text-secondary hover:text-primary transition-colors focus:ring-2 focus:ring-accent"
              >
                Score Another File
              </button>
              <button
                onClick={() => navigate('/')}
                className="flex-1 py-2.5 bg-accent hover:bg-accent-hover text-white text-xs font-semibold rounded-md shadow-xs transition-colors focus:ring-2 focus:ring-accent text-center"
              >
                View Prioritised Queue
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImportQueue;
