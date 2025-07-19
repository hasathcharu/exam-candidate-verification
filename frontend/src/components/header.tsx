import { HiOutlineLightBulb } from 'react-icons/hi';

export default function Header() {
  return (
    <div className='fixed top-5 z-50 flex align-center justify-left ml-5 w-full' >
      <div className='bg-background/20 bg-opacity-20 backdrop-filter backdrop-blur-xs border-gray-200 border rounded-[20px] p-1 shadow-md'>
        <a className='btn btn-ghost rounded-[16px] text-lg' href='/'>
          <HiOutlineLightBulb size={24} className='mb-[2px]' />Hikari Research
        </a>
      </div>
    </div>
  );
}
