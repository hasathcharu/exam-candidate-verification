import NvCard from '@/components/nv-card';
import AvCard from '@/components/av-card';
import Footer from '@/components/footer';
import Link from 'next/link';
import Section from '@/components/ui/section';

export default function App() {
  return (
    <div className='flex flex-col h-screen'>
      {/* Main content */}
      <div className="fixed inset-0 bg-[url('/background.svg')] bg-cover opacity-90 bg-center bg-no-repeat -z-1" />
      <main className='flex-grow'>
        <Section>
          <div className='hero h-[80vh] mt-[7vh]'>
            <div className='hero-content text-center'>
              <div className='max-w-[80ch]'>
                <h1 className='text-5xl font-bold'>
                  Writer Verification System
                </h1>
                <p className='py-6'>
                  Welcome to our Writer Verification Service, designed and
                  developed by Hikari Research. Pick your preferred verification
                  mode and we can get started!
                </p>
                <Link href='/quick-verification'>
                  <button className='btn btn-primary btn-lg'>
                    Get Started
                  </button>
                </Link>
              </div>
              
            </div>
            
          </div>
        </Section>
      </main>
      <Footer />
    </div>
  );
}
