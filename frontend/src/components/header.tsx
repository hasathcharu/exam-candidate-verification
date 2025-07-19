import { HiOutlineLightBulb } from 'react-icons/hi';

export default function Header() {
  return (
    <div className='fixed top-5 z-50 flex align-center justify-center w-full' >
      <div className='bg-background/20 bg-opacity-20 backdrop-filter backdrop-blur-xs border-gray-200 border rounded-[20px] p-2 shadow-md'>
        <a className='btn btn-ghost rounded-[12px] text-xl' href='/'>
          <HiOutlineLightBulb size={28} className='mb-[2px]' /> Home
        </a>
      </div>
    </div>
  );
}
