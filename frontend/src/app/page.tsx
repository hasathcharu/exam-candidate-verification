import NvCard from '@/components/nv-card';
import AvCard from '@/components/av-card';
import Footer from '@/components/footer';

export default function App() {
  return (
    <div className='flex flex-col h-screen'>
      {/* Main content */}
      <div className="fixed inset-0 bg-[url('/background.png')] bg-cover bg-center bg-no-repeat opacity-40" />
      <main className='flex-grow'>
        <div className='hero h-full'>
          <div className='hero-content text-center'>
            <div className='max-w-min'>
              <h1 className='text-5xl font-bold'>Writer Verification System</h1>
              <p className='py-6'>
                Welcome to our Writer Verification Service, designed and
                developed by Hikari Research. Pick your preferred verification
                mode and we can get started!
              </p>
              <div className='flex gap-10'>
                <NvCard />
                <div className='divider divider-horizontal'></div>
                <AvCard />
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
