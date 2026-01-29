import { Box, Mountain, Waves, Layers, ArrowRightLeft } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

const testCases = [
  {
    icon: Box,
    title: '3D Explosion Simulation',
    description:
      'Full three-dimensional simulation of explosive gas-solid dispersion, capturing complex shock-particle interactions and turbulent mixing processes.',
    tags: ['3D', 'Explosive', 'Turbulent'],
    specs: { dimensions: '3D', phases: 'Gas + 2 Solids', complexity: 'High' },
  },
  {
    icon: Mountain,
    title: '2D Explosion on Flat Topography',
    description:
      'Two-dimensional explosive flow simulation over flat terrain, ideal for validation studies and parameter sensitivity analysis.',
    tags: ['2D', 'Flat Terrain', 'Validation'],
    specs: { dimensions: '2D', phases: 'Gas + 2 Solids', complexity: 'Medium' },
  },
  {
    icon: Waves,
    title: 'Dilute Flow over Wavy Surface',
    description:
      'Simulation of dilute particle-laden flow over undulating topography, studying particle deposition and resuspension patterns.',
    tags: ['2D', 'Wavy Surface', 'Dilute'],
    specs: { dimensions: '2D', phases: 'Gas + 1 Solid', complexity: 'Medium' },
  },
  {
    icon: Layers,
    title: 'Fluidized Bed with Two Phases',
    description:
      'Bubbling fluidized bed simulation with two distinct solid phases, demonstrating particle segregation and mixing behavior.',
    tags: ['2D', 'Fluidized Bed', 'Dense'],
    specs: { dimensions: '2D', phases: 'Gas + 2 Solids', complexity: 'High' },
  },
  {
    icon: ArrowRightLeft,
    title: 'Impinging Flow with Two Phases',
    description:
      'Two-phase jet impingement simulation, analyzing particle impact dynamics and wall interaction effects.',
    tags: ['2D', 'Jet Flow', 'Impact'],
    specs: { dimensions: '2D', phases: 'Gas + 2 Solids', complexity: 'Medium' },
  },
];

export function TestCases() {
  return (
    <section id="test-cases" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <span className="text-orange-500 text-sm font-semibold tracking-wider uppercase">
            Test Cases
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mt-4 mb-6">
            Ready-to-Run
            <span className="text-gradient"> Examples</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto">
            Five comprehensive test cases are included to demonstrate OpenPDAC capabilities 
            and provide starting points for your own simulations.
          </p>
        </div>

        {/* Test Cases Grid */}
        <div className="grid lg:grid-cols-2 gap-6">
          {testCases.map((testCase, index) => (
            <div
              key={index}
              className={`group p-6 rounded-2xl glass hover:bg-white/[0.08] transition-all duration-300 ${
                index === 0 ? 'lg:col-span-2' : ''
              }`}
            >
              <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                {/* Icon */}
                <div className="w-14 h-14 rounded-xl bg-orange-500/10 flex items-center justify-center flex-shrink-0 group-hover:bg-orange-500/20 transition-colors">
                  <testCase.icon className="w-7 h-7 text-orange-500" />
                </div>

                {/* Content */}
                <div className="flex-1">
                  <div className="flex flex-wrap gap-2 mb-3">
                    {testCase.tags.map((tag, tagIndex) => (
                      <Badge
                        key={tagIndex}
                        variant="secondary"
                        className="bg-orange-500/10 text-orange-400 border-orange-500/20"
                      >
                        {tag}
                      </Badge>
                    ))}
                  </div>

                  <h3 className="text-xl font-semibold text-white mb-2">{testCase.title}</h3>
                  <p className="text-gray-400 mb-4">{testCase.description}</p>

                  {/* Specs */}
                  <div className="flex flex-wrap gap-4 text-sm">
                    {Object.entries(testCase.specs).map(([key, value]) => (
                      <div key={key} className="flex items-center gap-2">
                        <span className="text-gray-500 capitalize">{key}:</span>
                        <span className="text-gray-300">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Note */}
        <div className="mt-12 text-center">
          <p className="text-gray-500 text-sm">
            All test cases include complete setup files, mesh configurations, and post-processing scripts.
          </p>
        </div>
      </div>
    </section>
  );
}
