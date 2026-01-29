import { Navigation } from './sections/Navigation';
import { Hero } from './sections/Hero';
import { About } from './sections/About';
import { Features } from './sections/Features';
import { TestCases } from './sections/TestCases';
import { Installation } from './sections/Installation';
import { Footer } from './sections/Footer';

function App() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main>
        <Hero />
        <About />
        <Features />
        <TestCases />
        <Installation />
      </main>
      <Footer />
    </div>
  );
}

export default App;
