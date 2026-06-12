import { NavLink } from 'react-router-dom';
import { LayoutDashboard, PlusCircle, Calculator, FlaskConical } from 'lucide-react';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/experiments/new', label: 'Create Experiment', icon: PlusCircle },
  { to: '/power', label: 'Power Calculator', icon: Calculator },
];

export function Sidebar() {
  return (
    <aside className="w-60 bg-slate-900 text-slate-300 flex flex-col min-h-screen fixed left-0 top-0">
      <div className="flex items-center gap-2 px-5 py-5 border-b border-slate-700">
        <FlaskConical className="w-6 h-6 text-indigo-400" />
        <span className="text-lg font-semibold text-white">Experimentor</span>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <Icon className="w-5 h-5" />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="px-5 py-4 border-t border-slate-700 text-xs text-slate-500">
        A/B Testing Platform v0.1
      </div>
    </aside>
  );
}
