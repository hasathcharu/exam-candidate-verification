import Link from 'next/link';

export default function NvCard() {
  return (
    <div className='text-justify card w-82 bg-base-100 shadow-sm'>
      <div className='card-body'>
        {/* <span className='badge badge-xs badge-accent'>Fast & Scalable</span> */}
        <div className='flex justify-between'>
          <h2 className='text-2xl font-bold'>Quick Verification</h2>
        </div>
        <ul className='mt-2 flex flex-col gap-2 text-xs'>
          <li>
            <svg
              xmlns='http://www.w3.org/2000/svg'
              className='size-4 me-2 inline-block text-success'
              fill='none'
              viewBox='0 0 24 24'
              stroke='currentColor'
            >
              <path
                strokeLinecap='round'
                strokeLinejoin='round'
                strokeWidth='2'
                d='M5 13l4 4L19 7'
              />
            </svg>
            <span>Signature Forgery Detection</span>
          </li>
          <li>
            <svg
              xmlns='http://www.w3.org/2000/svg'
              className='size-4 me-2 inline-block text-success'
              fill='none'
              viewBox='0 0 24 24'
              stroke='currentColor'
            >
              <path
                strokeLinecap='round'
                strokeLinejoin='round'
                strokeWidth='2'
                d='M5 13l4 4L19 7'
              />
            </svg>
            <span>Handwriting Verification</span>
          </li>
          <li>
            <svg
              xmlns='http://www.w3.org/2000/svg'
              className='size-4 me-2 inline-block text-success'
              fill='none'
              viewBox='0 0 24 24'
              stroke='currentColor'
            >
              <path
                strokeLinecap='round'
                strokeLinejoin='round'
                strokeWidth='2'
                d='M5 13l4 4L19 7'
              />
            </svg>
            <span>Preliminary Verification</span>
          </li>
          <li>
            <svg
              xmlns='http://www.w3.org/2000/svg'
              className='size-4 me-2 inline-block text-success'
              fill='none'
              viewBox='0 0 24 24'
              stroke='currentColor'
            >
              <path
                strokeLinecap='round'
                strokeLinejoin='round'
                strokeWidth='2'
                d='M5 13l4 4L19 7'
              />
            </svg>
            <span>Great for most tasks</span>
          </li>
          <li>
            <svg
              xmlns='http://www.w3.org/2000/svg'
              className='size-4 me-2 inline-block text-success'
              fill='none'
              viewBox='0 0 24 24'
              stroke='currentColor'
            >
              <path
                strokeLinecap='round'
                strokeLinejoin='round'
                strokeWidth='2'
                d='M5 13l4 4L19 7'
              />
            </svg>
            <span>Minimal sample requirement</span>
          </li>
        </ul>
        <div className='mt-2'>
          <Link href='/quick-verification'>
            <button className='btn btn-primary btn-block text-lg rounded-lg'>Get Started</button>
          </Link>
        </div>
      </div>
    </div>
  );
}
