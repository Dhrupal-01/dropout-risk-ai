import React, { useState, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { AlertCircle, ArrowLeft, CheckCircle2, FileText, RefreshCw, Upload } from 'lucide-react';
import { uploadBatchCsv } from '../api/endpoints';
import { TIERS } from '../app/tiers';
import { formatCount } from '../app/format';
import AdminTokenNotice from '../components/AdminTokenNotice';
import TierLabel from '../components/TierLabel';

const RESULT_COUNT_FIELD = { High: 'high_risk_count', Medium: 'medium_risk_count', Low: 'low_risk_count' };

const ImportQueue = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [fileError, setFileError] = useState('');

  // Batch CSV scoring; afterwards every cached list and count is stale.
  const uploadMutation = useMutation({
    mutationFn: (uploadFile) => uploadBatchCsv(uploadFile, true),
    onSuccess: () => queryClient.invalidateQueries(),
  });

  const chooseFile = (candidate) => {
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith('.csv')) {
      setFileError('Please choose a .csv file.');
      return;
    }
    setFileError('');
    setFile(candidate);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    chooseFile(e.dataTransfer.files?.[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) uploadMutation.mutate(file);
  };

  const reset = () => {
    setFile(null);
    setFileError('');
    uploadMutation.reset();
  };

  const result = uploadMutation.data;

  return (
    <div className="px-4 sm:px-6 py-8 max-w-2xl space-y-6">
      <Link to="/app/students" className="inline-flex items-center gap-1.5 text-15 text-ink hover:underline">
        <ArrowLeft className="w-4 h-4" aria-hidden="true" />
        All students
      </Link>

      <div>
        <h1 className="font-display font-medium text-32 tracking-display text-graphite">Import students</h1>
        <p className="mt-2 max-w-measure text-15 text-slate">
          Upload a CSV of the records the college already keeps. Every row is scored, and the student list and
          overview update straight away.
        </p>
      </div>

      <section className="rounded-panel border border-rule bg-paper p-4 sm:p-5 space-y-4">
        {!result && (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              className={`flex flex-col items-center justify-center gap-3 rounded-panel border-2 border-dashed p-8 text-center transition-colors ${
                dragActive ? 'border-ink bg-ink-wash' : 'border-control'
              }`}
            >
              <input
                ref={fileInputRef}
                id="csv-file"
                type="file"
                accept=".csv"
                onChange={(e) => chooseFile(e.target.files?.[0])}
                className="peer sr-only"
              />
              <Upload className="w-7 h-7 text-slate" aria-hidden="true" />
              {file ? (
                <p className="flex items-center gap-2 text-15 text-graphite">
                  <FileText className="w-4 h-4 shrink-0 text-ink" aria-hidden="true" />
                  <span className="truncate max-w-[16rem]">{file.name}</span>
                  <span className="text-13 tabular-nums text-slate">({(file.size / 1024).toFixed(1)} KB)</span>
                </p>
              ) : (
                <p className="text-15 text-graphite">Drag a CSV file here, or</p>
              )}
              {/* The input is visually hidden, so the label shows its keyboard focus (same ring as *:focus-visible). */}
              <label
                htmlFor="csv-file"
                className="btn btn-secondary cursor-pointer peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-ink"
              >
                {file ? 'Choose a different file' : 'Choose a file'}
              </label>
            </div>

            {fileError && (
              <p role="alert" className="text-15 text-graphite">
                {fileError}
              </p>
            )}

            <p className="text-13 text-slate">
              The file needs a <code className="rounded-[4px] bg-ink-wash px-1 text-graphite">student_id</code> column and the
              raw columns of the feature contract (attendance, marks, learning-platform activity, fees). Label, engineered
              and protected columns are ignored, never scored.
            </p>

            <AdminTokenNotice />

            <button type="submit" disabled={!file || uploadMutation.isPending} className="btn btn-primary w-full disabled:opacity-50">
              {uploadMutation.isPending && <RefreshCw className="w-4 h-4 motion-safe:animate-spin" aria-hidden="true" />}
              {uploadMutation.isPending ? 'Scoring…' : 'Score students'}
            </button>
          </form>
        )}

        {uploadMutation.isError && (
          <div role="alert" className="space-y-3 rounded-control border border-rule p-4">
            <p className="flex items-start gap-2 text-15 text-graphite">
              <AlertCircle className="mt-0.5 w-4 h-4 shrink-0 text-slate" aria-hidden="true" />
              <span>
                <span className="font-semibold">The file could not be scored.</span>{' '}
                {uploadMutation.error?.message || 'Check the column names and value ranges.'}
              </span>
            </p>
            <button type="button" onClick={reset} className="btn btn-secondary">
              Try another file
            </button>
          </div>
        )}

        {result && (
          <div className="space-y-5">
            <p className="flex items-center gap-2 text-17 font-semibold text-graphite">
              <CheckCircle2 className="w-5 h-5 text-ink" aria-hidden="true" />
              {formatCount(result.total_students_evaluated ?? result.results?.length ?? 0)} students scored
            </p>

            <ul className="divide-y divide-rule rounded-control border border-rule">
              {TIERS.map((tier) => (
                <li key={tier.api} className="flex items-center justify-between px-4 py-2.5 text-15">
                  <TierLabel tier={tier.api} />
                  <span className="tabular-nums text-graphite">{formatCount(result[RESULT_COUNT_FIELD[tier.api]] ?? 0)}</span>
                </li>
              ))}
            </ul>

            {result.model_version && <p className="text-13 text-slate">Model {result.model_version}</p>}

            <div className="flex flex-wrap gap-3">
              <button type="button" onClick={reset} className="btn btn-secondary flex-1">
                Import another file
              </button>
              <button type="button" onClick={() => navigate('/app/students')} className="btn btn-primary flex-1">
                Open the student list
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export default ImportQueue;
