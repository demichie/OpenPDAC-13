import { Github, BookOpen, ExternalLink } from 'lucide-react';

const footerLinks = {
  project: [
    { name: 'GitHub Repository', href: 'https://github.com/demichie/OpenPDAC-13', icon: Github },
    { name: 'Releases', href: 'https://github.com/demichie/OpenPDAC-13/releases', icon: ExternalLink },
    { name: 'Issues', href: 'https://github.com/demichie/OpenPDAC-13/issues', icon: ExternalLink },
  ],
  resources: [
    { name: 'OpenFOAM Documentation', href: 'https://doc.openfoam.com/', icon: BookOpen },
    { name: 'OpenFOAM Wiki', href: 'https://wiki.openfoam.com/', icon: BookOpen },
    { name: 'CFD Online', href: 'https://www.cfd-online.com/', icon: ExternalLink },
  ],
};

export function Footer() {
  return (
    <footer className="relative py-16 border-t border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center">
                <span className="text-white font-bold">OP</span>
              </div>
              <span className="text-2xl font-bold text-white">OpenPDAC</span>
            </div>
            <p className="text-gray-400 mb-6 max-w-md">
              An open-source OpenFOAM module for simulating pyroclastic density currents 
              and granular flows with multiple dispersed solid phases.
            </p>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/demichie/OpenPDAC-13"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-all"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="https://doi.org/10.5281/zenodo.17054990"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-all"
              >
                <BookOpen className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Project Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Project</h3>
            <ul className="space-y-3">
              {footerLinks.project.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                  >
                    <link.icon className="w-4 h-4" />
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-white font-semibold mb-4">Resources</h3>
            <ul className="space-y-3">
              {footerLinks.resources.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                  >
                    <link.icon className="w-4 h-4" />
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-gray-500 text-sm">
            © {new Date().getFullYear()} OpenPDAC. Licensed under GPL-3.0.
          </p>
          <p className="text-gray-600 text-sm">
            Not approved or endorsed by OpenFOAM Foundation or ESI Ltd.
          </p>
        </div>
      </div>
    </footer>
  );
}
