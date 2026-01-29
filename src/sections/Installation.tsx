import { Terminal, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

const installSteps = [
  {
    title: 'Install OpenFOAM 13',
    description: 'OpenPDAC requires OpenFOAM 13 to be installed on your system.',
    code: `# Download and install OpenFOAM 13
wget https://dl.openfoam.com/ubuntu/openfoam13_20250911_amd64.deb
sudo apt install ./openfoam13_20250911_amd64.deb`,
  },
  {
    title: 'Clone the Repository',
    description: 'Download the OpenPDAC source code from GitHub.',
    code: `git clone https://github.com/demichie/OpenPDAC-13.git
cd OpenPDAC-13`,
  },
  {
    title: 'Set Environment',
    description: 'Source the OpenFOAM environment variables.',
    code: `source /usr/lib/openfoam/openfoam13/etc/bashrc`,
  },
  {
    title: 'Compile OpenPDAC',
    description: 'Build the solver and libraries.',
    code: `./Allwmake`,
  },
];

const dependencies = [
  'OpenFOAM 13 (openfoam13 package)',
  'Ubuntu 20.04+ or compatible Linux distribution',
  'GCC compiler (version 9 or higher)',
  'MPI library (OpenMPI recommended)',
  'CMake 3.15+',
];

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative mt-4 rounded-xl overflow-hidden bg-[#1a1a2e] border border-white/10">
      <div className="flex items-center justify-between px-4 py-2 bg-white/5 border-b border-white/10">
        <span className="text-xs text-gray-500">bash</span>
        <button
          onClick={copyToClipboard}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
        >
          {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto">
        <code className="text-sm text-gray-300 font-mono whitespace-pre">{code}</code>
      </pre>
    </div>
  );
}

export function Installation() {
  return (
    <section id="installation" className="relative py-24 lg:py-32 bg-gradient-to-b from-white/[0.02] to-transparent">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <span className="text-orange-500 text-sm font-semibold tracking-wider uppercase">
            Installation
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mt-4 mb-6">
            Get Started in
            <span className="text-gradient"> Minutes</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto">
            Follow these simple steps to install and run OpenPDAC on your system.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Installation Steps */}
          <div className="lg:col-span-2 space-y-6">
            {installSteps.map((step, index) => (
              <div key={index} className="p-6 rounded-2xl glass">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-orange-500/20 flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-semibold text-orange-500">{index + 1}</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-white mb-1">{step.title}</h3>
                    <p className="text-gray-400 text-sm mb-2">{step.description}</p>
                    <CodeBlock code={step.code} />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Dependencies */}
            <div className="p-6 rounded-2xl glass">
              <h3 className="text-lg font-semibold text-white mb-4">Dependencies</h3>
              <ul className="space-y-3">
                {dependencies.map((dep, index) => (
                  <li key={index} className="flex items-start gap-3 text-sm text-gray-400">
                    <div className="w-1.5 h-1.5 rounded-full bg-orange-500 mt-1.5 flex-shrink-0" />
                    {dep}
                  </li>
                ))}
              </ul>
            </div>

            {/* Quick Links */}
            <div className="p-6 rounded-2xl glass">
              <h3 className="text-lg font-semibold text-white mb-4">Quick Links</h3>
              <div className="space-y-3">
                <a
                  href="https://github.com/demichie/OpenPDAC-13"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
                >
                  <Terminal className="w-4 h-4" />
                  View on GitHub
                </a>
                <a
                  href="https://github.com/demichie/OpenPDAC-13/releases"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
                >
                  <Terminal className="w-4 h-4" />
                  Latest Releases
                </a>
              </div>
            </div>

            {/* Help */}
            <div className="p-6 rounded-2xl bg-orange-500/10 border border-orange-500/20">
              <h3 className="text-lg font-semibold text-white mb-2">Need Help?</h3>
              <p className="text-sm text-gray-400 mb-4">
                Check the GitHub repository for detailed documentation and community support.
              </p>
              <a
                href="https://github.com/demichie/OpenPDAC-13/issues"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" className="w-full border-orange-500/30 text-orange-400 hover:bg-orange-500/10">
                  Open an Issue
                </Button>
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
