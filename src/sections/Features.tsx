import { Check, Cpu, Globe, Shield, Workflow } from 'lucide-react';

const capabilities = [
  {
    icon: Cpu,
    title: 'Advanced Numerics',
    description: 'High-resolution schemes and robust solvers for stiff multiphase equations.',
    items: ['Kinetic theory closure', 'Frictional stress models', 'Particle pressure coupling'],
  },
  {
    icon: Workflow,
    title: 'Flexible Workflow',
    description: 'Seamless integration with OpenFOAM toolchain and pre/post-processing utilities.',
    items: ['blockMesh support', 'snappyHexMesh compatible', 'ParaView visualization'],
  },
  {
    icon: Globe,
    title: 'Large Scale Ready',
    description: 'Optimized for large domain simulations with proper boundary condition handling.',
    items: ['Hydrostatic initialization', 'Inflow/outflow BCs', 'Topography support'],
  },
  {
    icon: Shield,
    title: 'Open Source',
    description: 'GPL-3.0 licensed, free to use, modify, and distribute for research and commercial applications.',
    items: ['Full source access', 'Active development', 'Community driven'],
  },
];

export function Features() {
  return (
    <section id="features" className="relative py-24 lg:py-32 bg-gradient-to-b from-transparent to-white/[0.02]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <span className="text-orange-500 text-sm font-semibold tracking-wider uppercase">
            Features
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mt-4 mb-6">
            Built for
            <span className="text-gradient"> Scientific Rigor</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto">
            OpenPDAC combines cutting-edge numerical methods with practical engineering 
            capabilities to deliver accurate and efficient multiphase flow simulations.
          </p>
        </div>

        {/* Capabilities Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {capabilities.map((capability, index) => (
            <div
              key={index}
              className="p-8 rounded-2xl glass hover:bg-white/[0.08] transition-all duration-300"
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-orange-500/10 flex items-center justify-center flex-shrink-0">
                  <capability.icon className="w-6 h-6 text-orange-500" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white mb-2">{capability.title}</h3>
                  <p className="text-gray-400 mb-4">{capability.description}</p>
                  <ul className="space-y-2">
                    {capability.items.map((item, itemIndex) => (
                      <li key={itemIndex} className="flex items-center gap-2 text-sm text-gray-300">
                        <Check className="w-4 h-4 text-orange-500" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Technical Specs */}
        <div className="mt-16 grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { label: 'Base Package', value: 'openfoam13' },
            { label: 'Language', value: 'C/C++' },
            { label: 'Parallel', value: 'MPI Support' },
            { label: 'Platform', value: 'Linux/Ubuntu' },
          ].map((spec, index) => (
            <div key={index} className="text-center p-6 rounded-xl glass">
              <div className="text-sm text-gray-500 mb-1">{spec.label}</div>
              <div className="text-lg font-semibold text-white">{spec.value}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
