import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import OverviewCards from './components/OverviewCards';
import SceneExplorer from './components/SceneExplorer';
import AnalyticsCharts from './components/AnalyticsCharts';
import SpeakersGrid from './components/SpeakersGrid';
import TopicsEntities from './components/TopicsEntities';
import JsonViewer from './components/JsonViewer';

import {
  Film,
  Search,
  Upload,
  FileText,
  Play,
  Loader2,
  AlertCircle,
  TrendingUp,
  Users,
  Tag,
  FileJson,
} from 'lucide-react';

export default function App() {
  const [mode, setMode] = useState('corpus'); // 'corpus', 'upload', 'raw'
  const [searchQuery, setSearchQuery] = useState('');
  const [scriptsList, setScriptsList] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [rawText, setRawText] = useState('');
  const [rawTitle, setRawTitle] = useState('Custom Script');
  const [file, setFile] = useState(null);

  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiConnected, setApiConnected] = useState(false);
  const [activeTab, setActiveTab] = useState('scenes');

  // Check API Health & Fetch Scripts on mount
  useEffect(() => {
    fetchHealth();
    fetchScripts('');
  }, []);

  // Fetch scripts when search query changes
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchScripts(searchQuery);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      setApiConnected(res.ok);
    } catch (err) {
      setApiConnected(false);
    }
  };

  const fetchScripts = async (query) => {
    try {
      const res = await fetch(`/api/scripts?query=${encodeURIComponent(query)}&limit=0`);
      if (res.ok) {
        const data = await res.json();
        setScriptsList(data.results || []);
        // Deliberately not auto-selecting the first match here -- searching
        // should show every matching result for the user to pick from, not
        // silently jump to one.
      }
    } catch (err) {
      console.error('Failed fetching scripts list:', err);
    }
  };

  const handleGenerate = async (movie = selectedMovie) => {
    if (!movie && mode === 'corpus') return;
    setLoading(true);
    setError(null);

    try {
      let res;
      if (mode === 'corpus' && movie) {
        res = await fetch('/api/tag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            imdb_id: movie.imdb_id,
            use_transformers: false,
            include_dialogue: true,
          }),
        });
      } else if (mode === 'upload' && file) {
        const formData = new FormData();
        formData.append('file', file);
        res = await fetch('/api/tag/upload?use_transformers=false', {
          method: 'POST',
          body: formData,
        });
      } else if (mode === 'raw' && rawText) {
        res = await fetch('/api/tag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: rawText,
            title: rawTitle || 'Custom Script',
            use_transformers: false,
            include_dialogue: true,
          }),
        });
      }

      if (res && res.ok) {
        const resultMeta = await res.json();
        setMeta(resultMeta);
        if (mode === 'upload' || mode === 'raw') {
          // Newly-tagged uploads/pastes get persisted server-side — refresh
          // the corpus list so they show up in the search dropdown too.
          fetchScripts(searchQuery);
        }
      } else {
        const errData = await res?.json().catch(() => ({}));
        setError(errData.detail || 'Failed to tag screenplay metadata.');
      }
    } catch (err) {
      setError(err.message || 'Network error connecting to API.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0E14] text-slate-100 font-sans pb-16">
      
      {/* Top Minimalist Header */}
      <Header
        mode={mode}
        setMode={setMode}
        apiConnected={apiConnected}
      />

      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        
        {/* Main Minimalist Control Area */}
        <div className="ui-card rounded-xl p-4 mb-6">
          {mode === 'corpus' && (
            <div className="flex flex-col gap-3">
              <div className="relative w-full">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-3" />
                <input
                  type="text"
                  placeholder="Search movie titles (2,800+ screenplays)..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-[#0B0E14] text-slate-100 text-xs rounded-lg pl-9 pr-3 py-2.5 border border-[#1E2638] focus:outline-none focus:border-blue-500"
                />
              </div>

              {searchQuery && (
                scriptsList.length > 0 ? (
                  <div className="max-h-72 overflow-y-auto rounded-lg border border-[#1E2638] divide-y divide-[#1E2638]">
                    {scriptsList.slice(0, 50).map((m) => (
                      <button
                        key={m.imdb_id}
                        onClick={() => {
                          setSelectedMovie(m);
                          handleGenerate(m);
                        }}
                        disabled={loading}
                        className={`w-full text-left px-3 py-2 text-xs hover:bg-[#1E2638] transition-colors cursor-pointer disabled:opacity-50 flex items-center justify-between gap-3 ${
                          selectedMovie?.imdb_id === m.imdb_id ? 'bg-[#1E2638]' : ''
                        }`}
                      >
                        <span className="text-slate-100 truncate">
                          {m.title || 'Untitled'} {m.year ? <span className="text-slate-500">({m.year})</span> : ''}
                        </span>
                        {m.genres?.length > 0 && (
                          <span className="text-slate-500 shrink-0 text-[11px]">{m.genres.join(', ')}</span>
                        )}
                      </button>
                    ))}
                    {scriptsList.length > 50 && (
                      <div className="px-3 py-2 text-[11px] text-slate-500">
                        +{scriptsList.length - 50} more matches — refine your search to narrow it down
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-xs text-slate-500 px-1">No matching titles found.</div>
                )
              )}

              {selectedMovie && (
                <div className="flex items-center gap-2 text-xs text-slate-400 px-1">
                  {loading ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-current shrink-0" />
                  )}
                  <span>
                    {loading ? 'Analyzing ' : 'Selected: '}
                    <span className="text-slate-200">{selectedMovie.title}</span>
                    {selectedMovie.year ? ` (${selectedMovie.year})` : ''}
                  </span>
                </div>
              )}
            </div>
          )}

          {mode === 'upload' && (
            <div className="flex flex-col sm:flex-row items-center gap-3">
              <input
                type="file"
                accept=".txt,.srt"
                onChange={(e) => setFile(e.target.files[0])}
                className="w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-medium file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
              />
              <button
                onClick={() => handleGenerate()}
                disabled={loading || !file}
                className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors disabled:opacity-50 shrink-0 cursor-pointer"
              >
                {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                Analyze Upload
              </button>
            </div>
          )}

          {mode === 'raw' && (
            <div className="space-y-3">
              <input
                type="text"
                placeholder="Script Title (e.g. Inception)"
                value={rawTitle}
                onChange={(e) => setRawTitle(e.target.value)}
                className="w-full bg-[#0B0E14] text-slate-100 text-xs rounded-lg px-3 py-2 border border-[#1E2638] focus:outline-none focus:border-blue-500"
              />
              <textarea
                rows={4}
                placeholder="Paste raw screenplay text here..."
                value={rawText}
                onChange={(e) => setRawText(e.target.value)}
                className="w-full bg-[#0B0E14] text-slate-100 text-xs rounded-lg p-3 border border-[#1E2638] focus:outline-none focus:border-blue-500 font-mono"
              />
              <div className="flex justify-end">
                <button
                  onClick={() => handleGenerate()}
                  disabled={loading || !rawText}
                  className="flex items-center gap-2 px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors disabled:opacity-50 cursor-pointer"
                >
                  {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                  Analyze Text
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Error Alert Banner */}
        {error && (
          <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-lg p-3 mb-6 flex items-center gap-2 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Essential Results View */}
        {meta && (
          <div className="space-y-5 animate-fade-in">
            {/* Top Overview Bar */}
            <OverviewCards meta={meta} />

            {/* Clean Minimalist Tabs Navigation */}
            <div className="flex items-center gap-1 border-b border-[#1E2638] pb-1">
              <button
                onClick={() => setActiveTab('scenes')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === 'scenes'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-[#121721]'
                }`}
              >
                <Film className="w-3.5 h-3.5" />
                Scenes ({meta.segments?.length || 0})
              </button>

              <button
                onClick={() => setActiveTab('speakers')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === 'speakers'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-[#121721]'
                }`}
              >
                <Users className="w-3.5 h-3.5" />
                Characters ({Array.isArray(meta.speakers) ? meta.speakers.length : Object.keys(meta.speakers || {}).length})
              </button>

              <button
                onClick={() => setActiveTab('analytics')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === 'analytics'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-[#121721]'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5" />
                Analytics & Tone
              </button>

              <button
                onClick={() => setActiveTab('topics')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === 'topics'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-[#121721]'
                }`}
              >
                <Tag className="w-3.5 h-3.5" />
                Topics
              </button>

              <button
                onClick={() => setActiveTab('json')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === 'json'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-[#121721]'
                }`}
              >
                <FileJson className="w-3.5 h-3.5" />
                JSON
              </button>
            </div>

            {/* Active View */}
            <div>
              {activeTab === 'scenes' && <SceneExplorer segments={meta.segments} />}
              {activeTab === 'speakers' && <SpeakersGrid speakers={meta.speakers} />}
              {activeTab === 'analytics' && <AnalyticsCharts segments={meta.segments} />}
              {activeTab === 'topics' && (
                <TopicsEntities topics={meta.overall?.topics} entities={meta.overall?.entities} />
              )}
              {activeTab === 'json' && (
                <JsonViewer meta={meta} title={meta.title} imdbId={meta.imdb_id} />
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
