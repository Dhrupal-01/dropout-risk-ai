import React, { useState, useEffect } from 'react';
import { MapPin, AlertTriangle, TrendingUp, BookOpen, Users, Compass, ChevronRight } from 'lucide-react';

const GeoAnalytics = () => {
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState('Bihar');
  const [districts, setDistricts] = useState([]);
  const [_loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/v1/analytics/geo/states')
      .then((res) => res.json())
      .then((data) => {
        setStates(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed loading states:', err);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (selectedState) {
      fetch(`http://127.0.0.1:8000/api/v1/analytics/geo/districts?state=${encodeURIComponent(selectedState)}`)
        .then((res) => res.json())
        .then((data) => setDistricts(data))
        .catch((err) => console.error('Failed loading districts:', err));
    }
  }, [selectedState]);

  return (
    <div className="container mx-auto px-6 py-8 space-y-8">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-sky-500 mb-1">
            <Compass className="w-4 h-4" />
            <span>Macro Socio-Economic &amp; Regional Stratification</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Pan-India Geographic Risk Intelligence
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            State-level educational benchmarks (UDISE+ / AISHE) and District Multidimensional Poverty Indices (NITI Aayog).
          </p>
        </div>
      </div>

      {/* 4 Summary Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-slate-800 p-5 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
          <div className="flex justify-between items-center text-slate-500 dark:text-slate-400 text-xs font-medium uppercase">
            <span>States Monitored</span>
            <MapPin className="w-4 h-4 text-sky-500" />
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-2">{states.length || 10} States</div>
          <div className="text-xs text-slate-500 mt-1">Stratified educational telemetry</div>
        </div>

        <div className="bg-white dark:bg-slate-800 p-5 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
          <div className="flex justify-between items-center text-slate-500 dark:text-slate-400 text-xs font-medium uppercase">
            <span>Highest State MPI Poverty</span>
            <AlertTriangle className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-2xl font-bold text-amber-600 dark:text-amber-400 mt-2">Bihar (33.8%)</div>
          <div className="text-xs text-slate-500 mt-1">Headcount ratio under deprivation</div>
        </div>

        <div className="bg-white dark:bg-slate-800 p-5 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
          <div className="flex justify-between items-center text-slate-500 dark:text-slate-400 text-xs font-medium uppercase">
            <span>Highest Higher Ed GER</span>
            <TrendingUp className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mt-2">Tamil Nadu (51.4%)</div>
          <div className="text-xs text-slate-500 mt-1">Gross Enrollment Ratio benchmark</div>
        </div>

        <div className="bg-white dark:bg-slate-800 p-5 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
          <div className="flex justify-between items-center text-slate-500 dark:text-slate-400 text-xs font-medium uppercase">
            <span>Aspirational Districts</span>
            <Users className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-2">112 Target Districts</div>
          <div className="text-xs text-slate-500 mt-1">NITI Aayog ADP vulnerability tagging</div>
        </div>
      </div>

      {/* Main Split: States Table (Left) + District Deep Dive (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* State Vulnerability Table */}
        <div className="lg:col-span-7 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 flex items-center space-x-2">
              <BookOpen className="w-4 h-4 text-sky-500" />
              <span>State Educational &amp; Socio-Economic Benchmarks</span>
            </h3>
            <span className="text-xs text-slate-400">Click a state to inspect districts</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 dark:bg-slate-900 text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-200 dark:border-slate-700">
                <tr>
                  <th className="px-4 py-3">State Name</th>
                  <th className="px-4 py-3">Literacy Rate</th>
                  <th className="px-4 py-3">Higher Ed GER</th>
                  <th className="px-4 py-3">MPI Poverty</th>
                  <th className="px-4 py-3">Secondary Dropout</th>
                  <th className="px-4 py-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                {states.map((st) => (
                  <tr
                    key={st.state_code}
                    onClick={() => setSelectedState(st.state_name)}
                    className={`cursor-pointer transition-colors ${
                      selectedState === st.state_name
                        ? 'bg-sky-50 dark:bg-sky-950/40 font-semibold'
                        : 'hover:bg-slate-50 dark:hover:bg-slate-700/50'
                    }`}
                  >
                    <td className="px-4 py-3 text-slate-900 dark:text-slate-100 flex items-center space-x-2">
                      <span className="text-xs px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-mono">
                        {st.state_code}
                      </span>
                      <span>{st.state_name}</span>
                    </td>
                    <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{st.literacy_rate}%</td>
                    <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{st.ger_higher_ed}%</td>
                    <td className="px-4 py-3">
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          st.mpi_poverty_pct > 20
                            ? 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
                            : st.mpi_poverty_pct > 10
                            ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300'
                            : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                        }`}
                      >
                        {st.mpi_poverty_pct}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{st.school_dropout_rate_secondary}%</td>
                    <td className="px-4 py-3 text-sky-600 dark:text-sky-400">
                      <ChevronRight className="w-4 h-4" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* District Detail Cards for Selected State */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-200 dark:border-slate-700">
              <div>
                <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Selected State</span>
                <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100">{selectedState}</h3>
              </div>
              <span className="text-xs px-2.5 py-1 bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300 rounded-full font-medium">
                {districts.length} Districts Surveyed
              </span>
            </div>

            <div className="space-y-3">
              {districts.map((d) => (
                <div
                  key={d.district_name}
                  className="p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 space-y-2"
                >
                  <div className="flex justify-between items-start">
                    <div className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                      {d.district_name}
                    </div>
                    {d.is_aspirational && (
                      <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 rounded-full font-bold">
                        ★ Aspirational District
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs pt-1 border-t border-slate-200/50 dark:border-slate-700/50 text-slate-600 dark:text-slate-300">
                    <div>
                      <span className="text-slate-400 block text-[10px]">MPI Poverty</span>
                      <span className="font-semibold text-slate-800 dark:text-slate-200">{d.mpi_headcount_pct}%</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px]">Rurality</span>
                      <span className="font-semibold text-slate-800 dark:text-slate-200">{d.rurality_pct}%</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px]">Pupil-Teacher (PTR)</span>
                      <span className="font-semibold text-slate-800 dark:text-slate-200">{d.pupil_teacher_ratio}:1</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GeoAnalytics;
