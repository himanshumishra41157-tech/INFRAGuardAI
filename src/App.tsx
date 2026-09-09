import { useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  Building2,
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Download,
  Gauge,
  LayoutDashboard,
  MapPin,
  Menu,
  Network,
  PanelLeftClose,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  TrendingUp,
  X,
} from 'lucide-react';

type Role = 'National / MoSPI Officer' | 'Ministry Admin' | 'Public / Observer';
type Tab = 'triage' | 'predict' | 'simulate' | 'breakdown';
type RiskBand = 'High' | 'Medium' | 'Low';

type Project = {
  code: string;
  name: string;
  ministry: string;
  sector: string;
  state: string;
  cost: number;
  revised: number;
  spend: number;
  progress: number;
  delay: number;
  reason: string;
  risk: number;
  band: RiskBand;
};

const projects: Project[] = [
  { code: '701658', name: 'Eastern Freight Corridor — Phase II', ministry: 'Railways', sector: 'Rail Infrastructure', state: 'Uttar Pradesh', cost: 1180, revised: 1485, spend: 820, progress: 64, delay: 18, reason: 'Land Acquisition', risk: 78, band: 'High' },
  { code: '704219', name: 'Mumbai–Nagpur Expressway Link', ministry: 'Road Transport & Highways', sector: 'Highways', state: 'Maharashtra', cost: 2310, revised: 2685, spend: 1910, progress: 78, delay: 9, reason: 'Utility Shifting', risk: 61, band: 'High' },
  { code: '709341', name: 'Upper Siang Hydroelectric Project', ministry: 'Power', sector: 'Power Generation', state: 'Arunachal Pradesh', cost: 3420, revised: 3820, spend: 990, progress: 31, delay: 28, reason: 'Forest Clearances', risk: 74, band: 'High' },
  { code: '702776', name: 'Bengaluru Metro — Phase II Extension', ministry: 'Urban Affairs', sector: 'Urban Transit', state: 'Karnataka', cost: 940, revised: 1085, spend: 740, progress: 71, delay: 6, reason: 'Contracting Issue', risk: 43, band: 'Medium' },
  { code: '706513', name: 'Greenfield Terminal Expansion', ministry: 'Civil Aviation', sector: 'Aviation', state: 'Gujarat', cost: 640, revised: 702, spend: 570, progress: 83, delay: 3, reason: 'None', risk: 22, band: 'Low' },
  { code: '703887', name: 'BharatNet Rural Fibre Backbone', ministry: 'Telecommunications', sector: 'Telecom', state: 'Odisha', cost: 510, revised: 670, spend: 460, progress: 58, delay: 14, reason: 'Utility Shifting', risk: 56, band: 'Medium' },
  { code: '708042', name: 'Kandla Container Terminal Upgrade', ministry: 'Shipping', sector: 'Ports', state: 'Gujarat', cost: 780, revised: 840, spend: 745, progress: 91, delay: 2, reason: 'None', risk: 18, band: 'Low' },
  { code: '705628', name: 'Rajasthan Solar Park — Cluster 3', ministry: 'New & Renewable Energy', sector: 'Renewable Energy', state: 'Rajasthan', cost: 1760, revised: 1980, spend: 1160, progress: 49, delay: 11, reason: 'Contracting Issue', risk: 48, band: 'Medium' },
];

const tabs: { id: Tab; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'triage', label: 'Portfolio triage', icon: LayoutDashboard },
  { id: 'predict', label: 'Prediction & explainability', icon: Sparkles },
  { id: 'simulate', label: 'What-if simulator', icon: SlidersHorizontal },
  { id: 'breakdown', label: 'Breakdown & export', icon: BarChart3 },
];

const money = (value: number) => `₹${value.toLocaleString('en-IN')} Cr`;
const bandClass = (band: RiskBand) => band.toLowerCase();

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('triage');
  const [role, setRole] = useState<Role>('National / MoSPI Officer');
  const [ministry, setMinistry] = useState('All ministries');
  const [selectedCode, setSelectedCode] = useState(projects[0].code);
  const [search, setSearch] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [progressBoost, setProgressBoost] = useState(8);
  const [milestonesRecovered, setMilestonesRecovered] = useState(2);
  const [bottleneckCleared, setBottleneckCleared] = useState(true);

  const visibleProjects = useMemo(() => projects.filter((project) => {
    const ministryMatch = ministry === 'All ministries' || project.ministry === ministry;
    const searchMatch = `${project.name} ${project.code} ${project.state}`.toLowerCase().includes(search.toLowerCase());
    return ministryMatch && searchMatch;
  }), [ministry, search]);

  const selectedProject = projects.find((project) => project.code === selectedCode) ?? projects[0];
  const simulatedRisk = Math.max(0, Math.round(selectedProject.risk - progressBoost * 1.7 - milestonesRecovered * 2.5 - (bottleneckCleared ? 7 : 0)));
  const riskDelta = simulatedRisk - selectedProject.risk;
  const highRisk = visibleProjects.filter((project) => project.band === 'High').length;
  const totalRevised = visibleProjects.reduce((sum, project) => sum + project.revised, 0);
  const totalSpend = visibleProjects.reduce((sum, project) => sum + project.spend, 0);

  const openProject = (code: string, tab: Tab = 'predict') => {
    setSelectedCode(code);
    setActiveTab(tab);
  };

  const exportCsv = () => {
    const headers = ['Project code', 'Project name', 'Ministry', 'Sector', 'State', 'Revised cost (Cr)', 'Progress (%)', 'Delay (months)', 'Risk score', 'Risk band'];
    const rows = visibleProjects.map((project) => [project.code, project.name, project.ministry, project.sector, project.state, project.revised, project.progress, project.delay, project.risk, project.band]);
    const csv = [headers, ...rows].map((row) => row.map((cell) => `"${String(cell).split('"').join('""')}"`).join(',')).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'infraguard-portfolio-report.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'is-open' : 'is-collapsed'}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><ShieldCheck size={22} strokeWidth={2.5} /></div>
          {sidebarOpen && <div><strong>INFRAguard</strong><span>AI MONITORING LAYER</span></div>}
        </div>
        {sidebarOpen && <div className="sidebar-section-label">Workspace</div>}
        <nav className="side-nav" aria-label="Dashboard sections">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button className={`nav-item ${activeTab === id ? 'active' : ''}`} key={id} onClick={() => setActiveTab(id)} title={label}>
              <Icon size={18} />{sidebarOpen && <span>{label}</span>}{sidebarOpen && id === 'triage' && <span className="nav-count">3</span>}
            </button>
          ))}
        </nav>
        {sidebarOpen && <>
          <div className="sidebar-section-label">Data context</div>
          <div className="context-card"><div className="context-icon"><Network size={16} /></div><div><strong>PAIMANA aligned</strong><span>April 2026 report context</span></div></div>
          <div className="context-card"><div className="context-icon blue"><Building2 size={16} /></div><div><strong>Central sector</strong><span>Projects above ₹150 Cr</span></div></div>
          <div className="sidebar-footer"><div className="avatar">MS</div><div><strong>MoSPI workspace</strong><span>Read-only prototype</span></div><ChevronDown size={15} /></div>
        </>}
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="Toggle sidebar">{sidebarOpen ? <PanelLeftClose size={19} /> : <Menu size={19} />}</button>
          <div className="breadcrumb"><span>INFRAguard AI</span><span>/</span><strong>{tabs.find((tab) => tab.id === activeTab)?.label}</strong></div>
          <div className="topbar-actions"><span className="live-status"><i /> Live portfolio</span><button className="icon-button" aria-label="Help"><CircleHelp size={18} /></button><button className="icon-button notification" aria-label="Notifications"><Bell size={18} /><b /></button></div>
        </header>

        <div className="page-wrap">
          <section className="hero-row">
            <div><div className="eyebrow"><span className="eyebrow-dot" /> Predictive intelligence layer</div><h1>Infrastructure, <em>made visible.</em></h1><p>Monitor cost, schedule, and delivery signals across the national project portfolio.</p></div>
            <div className="role-control"><label>Viewing as</label><select value={role} onChange={(event) => setRole(event.target.value as Role)}><option>National / MoSPI Officer</option><option>Ministry Admin</option><option>Public / Observer</option></select></div>
          </section>

          <section className="filter-bar">
            <div className="filter-search"><Search size={17} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search projects, codes, or states..." />{search && <button onClick={() => setSearch('')} aria-label="Clear search"><X size={15} /></button>}</div>
            <select value={ministry} onChange={(event) => setMinistry(event.target.value)}><option>All ministries</option><option>Railways</option><option>Road Transport & Highways</option><option>Power</option><option>Urban Affairs</option><option>Civil Aviation</option><option>Telecommunications</option><option>Shipping</option><option>New & Renewable Energy</option></select>
            <div className="filter-divider" /><span className="updated-label"><span /> Updated 12 min ago</span>
          </section>

          {role === 'Public / Observer' && <div className="observer-note"><ShieldCheck size={17} /><span><strong>Observer view</strong> Showing high-level portfolio signals only. Project-level interventions are read-only.</span></div>}

          {activeTab === 'triage' && <TriageView projects={visibleProjects} highRisk={highRisk} totalRevised={totalRevised} totalSpend={totalSpend} onOpenProject={openProject} />}
          {activeTab === 'predict' && <PredictView project={selectedProject} onSelect={setSelectedCode} />}
          {activeTab === 'simulate' && <SimulateView project={selectedProject} projects={visibleProjects} onSelect={setSelectedCode} progressBoost={progressBoost} setProgressBoost={setProgressBoost} milestonesRecovered={milestonesRecovered} setMilestonesRecovered={setMilestonesRecovered} bottleneckCleared={bottleneckCleared} setBottleneckCleared={setBottleneckCleared} simulatedRisk={simulatedRisk} riskDelta={riskDelta} />}
          {activeTab === 'breakdown' && <BreakdownView projects={visibleProjects} onExport={exportCsv} />}
        </div>
      </main>
    </div>
  );
}

function KpiCard({ label, value, detail, icon: Icon, tone, trend }: { label: string; value: string; detail: string; icon: typeof Gauge; tone: string; trend?: 'up' | 'down' }) {
  return <div className="kpi-card"><div className={`kpi-icon ${tone}`}><Icon size={18} /></div><div className="kpi-label">{label}</div><div className="kpi-value">{value}</div><div className="kpi-detail">{trend && (trend === 'up' ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />)}{detail}</div></div>;
}

function TriageView({ projects: visibleProjects, highRisk, totalRevised, totalSpend, onOpenProject }: { projects: Project[]; highRisk: number; totalRevised: number; totalSpend: number; onOpenProject: (code: string, tab?: Tab) => void }) {
  const maxRisk = Math.max(...visibleProjects.map((project) => project.risk), 1);
  return <>
    <div className="kpi-grid"><KpiCard label="Monitored projects" value={String(visibleProjects.length)} detail="Central sector portfolio" icon={Building2} tone="lavender" /><KpiCard label="High-risk attention" value={String(highRisk).padStart(2, '0')} detail="Requires intervention" icon={AlertTriangle} tone="coral" trend="up" /><KpiCard label="Revised portfolio" value={money(totalRevised)} detail="Across visible projects" icon={TrendingUp} tone="blue" /><KpiCard label="Aggregate spend" value={money(totalSpend)} detail={`${Math.round((totalSpend / totalRevised) * 100)}% of revised cost`} icon={Target} tone="mint" /></div>
    <div className="section-grid"><section className="panel chart-panel"><div className="panel-heading"><div><span className="section-kicker">Portfolio signal map</span><h2>Progress vs. expenditure</h2></div><span className="legend-chip"><i className="legend-gradient" /> Composite risk</span></div><div className="scatter-chart"><div className="y-label">EXPENDITURE %</div><div className="chart-grid"><span className="axis-mark y100">100</span><span className="axis-mark y50">50</span><span className="axis-mark y0">0</span><div className="diagonal" />{visibleProjects.map((project) => { const expenditure = Math.round((project.spend / project.revised) * 100); return <button key={project.code} className={`scatter-dot ${bandClass(project.band)}`} style={{ left: `${Math.min(project.progress, 96)}%`, bottom: `${Math.min(expenditure, 96)}%`, transform: `scale(${0.8 + project.risk / 180})` }} title={`${project.name}: risk ${project.risk}`} onClick={() => onOpenProject(project.code)} />; })}<div className="x-label">PHYSICAL PROGRESS % <span>0</span><span>50</span><span>100</span></div></div></div><div className="chart-note"><span className="trend-line" /> The diagonal represents spend aligned with physical progress. Points above it indicate spend ahead of delivery.</div></section><section className="panel signal-panel"><div className="panel-heading"><div><span className="section-kicker">Model watchlist</span><h2>Portfolio health</h2></div><Gauge size={19} className="muted-icon" /></div><div className="health-score"><div className="health-ring"><strong>68</strong><span>/ 100</span></div><div><strong>Needs attention</strong><p>3 signals have moved since last review.</p></div></div><div className="health-bar"><span style={{ width: '68%' }} /></div><div className="health-rows"><div><span><i className="dot coral" />Delivery risk</span><strong>Elevated</strong></div><div><span><i className="dot amber" />Cost pressure</span><strong>Watch</strong></div><div><span><i className="dot mint" />Data freshness</span><strong>Healthy</strong></div></div></section></div>
    <section className="panel warning-panel"><div className="panel-heading"><div><span className="section-kicker coral-text">Priority early warnings</span><h2>Where to look first</h2></div><button className="text-button" onClick={() => onOpenProject(visibleProjects[0]?.code ?? projects[0].code, 'breakdown')}>View portfolio <ArrowUpRight size={14} /></button></div><div className="table-wrap"><table><thead><tr><th>Project</th><th>Ministry</th><th>Signal</th><th>Progress</th><th>Risk score</th><th /></tr></thead><tbody>{[...visibleProjects].sort((a, b) => b.risk - a.risk).slice(0, 5).map((project) => <tr key={project.code}><td><button className="project-cell" onClick={() => onOpenProject(project.code)}><span className={`project-logo ${bandClass(project.band)}`}>{project.sector.slice(0, 1)}</span><span><strong>{project.name}</strong><small>{project.code} · {project.state}</small></span></button></td><td>{project.ministry}</td><td><span className={`status-pill ${bandClass(project.band)}`}><i />{project.reason === 'None' ? 'On track' : project.reason}</span></td><td><div className="progress-cell"><div className="mini-progress"><span style={{ width: `${project.progress}%` }} /></div><small>{project.progress}%</small></div></td><td><div className="risk-score"><strong>{project.risk}</strong><span className={`risk-badge ${bandClass(project.band)}`}>{project.band}</span></div></td><td><button className="row-arrow" onClick={() => onOpenProject(project.code)} aria-label={`Open ${project.name}`}><ArrowUpRight size={16} /></button></td></tr>)}</tbody></table>{visibleProjects.length === 0 && <div className="empty-state">No projects match the current filters.</div>}</div></section>
  </>;
}

function PredictView({ project, onSelect }: { project: Project; onSelect: (code: string) => void }) {
  const driverValues = [{ label: 'Cost overrun pressure', value: project.revised - project.cost, color: 'coral' }, { label: 'Schedule delay', value: project.delay * 2.2, color: 'amber' }, { label: 'Progress variance', value: 100 - project.progress, color: 'blue' }, { label: 'Milestone slippage', value: 48, color: 'lavender' }];
  return <><div className="detail-toolbar"><div><span className="section-kicker">Prediction engine</span><h2>Project-level intelligence</h2></div><select value={project.code} onChange={(event) => onSelect(event.target.value)}>{projects.map((item) => <option key={item.code} value={item.code}>{item.code} — {item.name}</option>)}</select></div><section className="project-hero panel"><div className="project-hero-main"><span className={`large-project-logo ${bandClass(project.band)}`}><Building2 size={24} /></span><div><span className="section-kicker">{project.code} · {project.sector}</span><h2>{project.name}</h2><p><MapPin size={14} /> {project.state} <span className="separator">·</span> {project.ministry}</p></div></div><span className={`status-pill ${bandClass(project.band)}`}><i />{project.band} risk</span></section><div className="risk-grid"><RiskGauge label="Cost overrun" value={Math.round((project.revised - project.cost) / project.cost * 100)} tone="lavender" /><RiskGauge label="Schedule delay" value={Math.min(100, project.delay * 3)} tone="blue" /><RiskGauge label="Unified risk score" value={project.risk} tone="coral" /></div><div className="section-grid"><section className="panel attribution-panel"><div className="panel-heading"><div><span className="section-kicker">Explainability layer</span><h2>What is driving the risk?</h2></div><span className="shap-badge"><Sparkles size={13} /> SHAP attributions</span></div><p className="panel-intro">The model highlights the signals contributing most to this project’s current score.</p>{driverValues.map((driver, index) => <div className="driver-row" key={driver.label}><div className="driver-label"><span>{String(index + 1).padStart(2, '0')}</span><strong>{driver.label}</strong><b>{Math.round(driver.value)}%</b></div><div className="driver-track"><span className={driver.color} style={{ width: `${Math.min(100, Math.max(22, driver.value))}%` }} /></div></div>)}</section><section className="panel snapshot-panel"><div className="panel-heading"><div><span className="section-kicker">Project snapshot</span><h2>Current position</h2></div><CalendarClock size={19} className="muted-icon" /></div><div className="snapshot-list"><div><span>Physical progress</span><strong>{project.progress}%</strong></div><div><span>Revised cost</span><strong>{money(project.revised)}</strong></div><div><span>Cumulative spend</span><strong>{money(project.spend)}</strong></div><div><span>Expected completion</span><strong>{project.delay > 12 ? 'Q4 2027' : 'Q2 2027'}</strong></div></div><div className="snapshot-callout"><AlertTriangle size={16} /><span>Primary bottleneck: <strong>{project.reason}</strong></span></div></section></div></>;
}

function RiskGauge({ label, value, tone }: { label: string; value: number; tone: string }) {
  const circumference = 2 * Math.PI * 42;
  return <div className="risk-gauge panel"><div className="gauge-ring"><svg viewBox="0 0 100 100"><circle className="gauge-base" cx="50" cy="50" r="42" /><circle className={`gauge-progress ${tone}`} cx="50" cy="50" r="42" strokeDasharray={circumference} strokeDashoffset={circumference - (circumference * value) / 100} /></svg><div><strong>{value}</strong><span>/ 100</span></div></div><div><span className="gauge-label">{label}</span><span className={`gauge-status ${value >= 60 ? 'bad' : value >= 30 ? 'warn' : 'good'}`}>{value >= 60 ? 'Needs attention' : value >= 30 ? 'Monitor closely' : 'Within range'}</span></div></div>;
}

function SimulateView({ project, projects: visibleProjects, onSelect, progressBoost, setProgressBoost, milestonesRecovered, setMilestonesRecovered, bottleneckCleared, setBottleneckCleared, simulatedRisk, riskDelta }: { project: Project; projects: Project[]; onSelect: (code: string) => void; progressBoost: number; setProgressBoost: (value: number) => void; milestonesRecovered: number; setMilestonesRecovered: (value: number) => void; bottleneckCleared: boolean; setBottleneckCleared: (value: boolean) => void; simulatedRisk: number; riskDelta: number }) {
  return <><div className="detail-toolbar"><div><span className="section-kicker">Scenario planning</span><h2>Test an intervention</h2></div><select value={project.code} onChange={(event) => onSelect(event.target.value)}>{visibleProjects.map((item) => <option key={item.code} value={item.code}>{item.code} — {item.name}</option>)}</select></div><div className="section-grid simulate-layout"><section className="panel controls-panel"><div className="panel-heading"><div><span className="section-kicker">Intervention controls</span><h2>Change the conditions</h2></div><SlidersHorizontal size={19} className="muted-icon" /></div><label className="range-label"><span>Improve physical progress</span><strong>+{progressBoost}%</strong></label><input className="range-input" type="range" min="0" max="20" value={progressBoost} onChange={(event) => setProgressBoost(Number(event.target.value))} /><div className="range-scale"><span>Current {project.progress}%</span><span>+20%</span></div><label className="range-label"><span>Recover delayed milestones</span><strong>+{milestonesRecovered}</strong></label><input className="range-input blue-range" type="range" min="0" max="5" value={milestonesRecovered} onChange={(event) => setMilestonesRecovered(Number(event.target.value))} /><div className="range-scale"><span>Current 7 / 12</span><span>+5</span></div><button className={`toggle-row ${bottleneckCleared ? 'checked' : ''}`} onClick={() => setBottleneckCleared(!bottleneckCleared)}><span className="toggle-check">{bottleneckCleared && <CheckCircle2 size={17} />}</span><span><strong>Clear primary bottleneck</strong><small>{project.reason} resolution plan</small></span><span className="toggle-switch"><i /></span></button></section><section className="panel outcome-panel"><div className="panel-heading"><div><span className="section-kicker">Projected outcome</span><h2>Risk score movement</h2></div><span className="scenario-tag"><Sparkles size={13} /> Live model</span></div><div className="compare-score"><div><span>Current</span><strong>{project.risk}</strong><small className={`risk-badge ${bandClass(project.band)}`}>{project.band}</small></div><div className={`outcome-arrow ${riskDelta < 0 ? 'positive' : 'negative'}`}><ArrowDownRight size={22} /><span>{Math.abs(riskDelta)} pts</span></div><div><span>Simulated</span><strong className={riskDelta < 0 ? 'positive-text' : 'negative-text'}>{simulatedRisk}</strong><small className={`risk-badge ${simulatedRisk >= 60 ? 'high' : simulatedRisk >= 30 ? 'medium' : 'low'}`}>{simulatedRisk >= 60 ? 'High' : simulatedRisk >= 30 ? 'Medium' : 'Low'}</small></div></div><div className="outcome-bar"><div><span>Current risk</span><b style={{ width: `${project.risk}%` }} /></div><div><span>Simulated risk</span><b className="simulated" style={{ width: `${simulatedRisk}%` }} /></div></div><div className="recommendation"><Target size={17} /><div><strong>Recommended path</strong><p>{riskDelta < 0 ? 'This combination meaningfully reduces delivery risk. Prioritize the bottleneck resolution first.' : 'Increase the intervention intensity to move this project back into a safer range.'}</p></div></div></section></div></>;
}

function BreakdownView({ projects: visibleProjects, onExport }: { projects: Project[]; onExport: () => void }) {
  const sectorData = visibleProjects.reduce<Record<string, number>>((result, project) => { result[project.sector] = (result[project.sector] ?? 0) + 1; return result; }, {});
  const ministryData = visibleProjects.reduce<Record<string, number>>((result, project) => { result[project.ministry] = (result[project.ministry] ?? 0) + 1; return result; }, {});
  const colors = ['#8b7cf6', '#73b7f5', '#5ac7a7', '#f1b85b', '#ee8d99', '#8ba4c7'];
  return <><div className="detail-toolbar"><div><span className="section-kicker">Portfolio composition</span><h2>Breakdown & reporting</h2></div><button className="primary-button" onClick={onExport}><Download size={16} /> Export CSV report</button></div><div className="breakdown-grid"><section className="panel distribution-panel"><div className="panel-heading"><div><span className="section-kicker">By sector</span><h2>Project distribution</h2></div></div><div className="donut-layout"><div className="donut" style={{ background: `conic-gradient(${colors.map((color, index) => `${color} 0 ${(Object.values(sectorData).slice(0, index + 1).reduce((a, b) => a + b, 0) / visibleProjects.length) * 360}deg`).join(', ')})` }}><div><strong>{visibleProjects.length}</strong><span>projects</span></div></div><div className="chart-legend">{Object.entries(sectorData).map(([sector, count], index) => <div key={sector}><span><i style={{ background: colors[index % colors.length] }} />{sector}</span><strong>{count}</strong></div>)}</div></div></section><section className="panel distribution-panel"><div className="panel-heading"><div><span className="section-kicker">By ministry</span><h2>Portfolio footprint</h2></div></div><div className="bar-list">{Object.entries(ministryData).map(([name, count], index) => <div className="bar-item" key={name}><div><span>{name}</span><strong>{count}</strong></div><div className="bar-track"><i style={{ width: `${count / Math.max(...Object.values(ministryData)) * 100}%`, background: colors[index % colors.length] }} /></div></div>)}</div></section></div><section className="panel export-panel"><div className="panel-heading"><div><span className="section-kicker">Report preview</span><h2>Portfolio summary</h2></div><span className="record-count">{visibleProjects.length} records</span></div><div className="table-wrap"><table><thead><tr><th>Project</th><th>Ministry</th><th>Revised cost</th><th>Progress</th><th>Risk</th></tr></thead><tbody>{visibleProjects.map((project) => <tr key={project.code}><td><strong>{project.name}</strong><small>{project.code}</small></td><td>{project.ministry}</td><td>{money(project.revised)}</td><td>{project.progress}%</td><td><span className={`risk-badge ${bandClass(project.band)}`}>{project.risk} · {project.band}</span></td></tr>)}</tbody></table></div></section></>;
}

export default App;
