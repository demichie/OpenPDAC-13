import { Play, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/50 to-background" />
      
      {/* Animated Gradient Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-orange-500/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl animate-pulse delay-1000" />

      {/* Content */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center pt-20">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-sm text-gray-300">OpenFOAM Module for Multiphase Flows</span>
        </div>

        {/* Title */}
        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6">
          <span className="text-white">Open</span>
          <span className="text-gradient">PDAC</span>
        </h1>

        {/* Subtitle */}
        <p className="text-xl sm:text-2xl text-gray-400 mb-4 max-w-3xl mx-auto">
          Advanced Computational Fluid Dynamics for
          <span className="text-white"> Pyroclastic Density Currents</span> and
          <span className="text-white"> Granular Flows</span>
        </p>

        {/* Description */}
        <p className="text-gray-500 mb-12 max-w-2xl mx-auto">
          A powerful OpenFOAM module extending multiphaseEuler with kinetic theory for multiple 
          dispersed solid phases and Lagrangian particle tracking.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
          <a href="#installation">
            <Button
              size="lg"
              className="bg-orange-500 hover:bg-orange-600 text-white px-8 py-6 text-lg glow-orange"
            >
              <Download className="w-5 h-5 mr-2" />
              Get Started
            </Button>
          </a>
          <a href="#test-cases">
            <Button
              size="lg"
              variant="outline"
              className="border-white/20 text-white hover:bg-white/10 px-8 py-6 text-lg"
            >
              <Play className="w-5 h-5 mr-2" />
              View Examples
            </Button>
          </a>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-3xl mx-auto">
          {[
            { value: '5+', label: 'Test Cases' },
            { value: '262+', label: 'Commits' },
            { value: 'C/C++', label: 'Core Language' },
            { value: 'GPL-3', label: 'License' },
          ].map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-2xl sm:text-3xl font-bold text-gradient">{stat.value}</div>
              <div className="text-sm text-gray-500">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2">
        <div className="w-6 h-10 rounded-full border-2 border-white/20 flex justify-center pt-2">
          <div className="w-1 h-2 bg-orange-500 rounded-full animate-bounce" />
        </div>
      </div>
    </section>
  );
}
