import { useEffect, useState } from 'react';
export default function Home({
  result,
  confidence,
  type,
}: {
  result: boolean;
  confidence: number;
  type: 'module1' | 'module2' | 'module3';
}) {
  const card_confidence = confidence;
  let text: string;
  let title: string;
  switch (type) {
    case 'module1':
      title = 'Signature Forgery Detection';
      text = result ? 'Genuine Signature' : 'Forged Signature';
      break;
    case 'module2':
      title = 'Quick Handwriting Verification';
      text = result ? 'Same Writer' : 'Different Writer';
      break;
    default:
      title = 'Personalized Handwriting Verification';
      text = result ? 'Same Writer' : 'Different Writer';
      break;
  }
  const CONFIDENCE_THRESHOLD = parseFloat(
    process.env.NEXT_PUBLIC_CONFIDENCE_THRESHOLD || '0.7'
  );
  let color = 'success';
  if (confidence < CONFIDENCE_THRESHOLD) {
    color = 'warning';
  }
  if (!result) {
    color = 'error';
  }

  switch (color) {
    case 'error':
      return (
        <div className='card bg-[var(--color-error-content)] flex-1'>
          <div className='card-body flex-1 flex flex-row items-center gap-4 justify-between'>
            <div>
              <h2 className='card-title flex-1'>{title}</h2>
              <h3 className='card-title text-sm flex-1'>{text}</h3>
            </div>

            <div className='flex flex-col items-center gap-2 text-sm'>
              <div
                className='radial-progress text-[var(--color-error)] text-xs font-bold'
                style={
                  {
                    '--value': `${card_confidence * 100}`,
                    '--size': '3.5rem',
                    '--thickness': '0.35rem',
                  } as React.CSSProperties
                }
                aria-valuenow={card_confidence * 100}
                role='progressbar'
              >
                {Math.round(card_confidence * 100)}%
              </div>
              <span>Confidence</span>
            </div>
          </div>
        </div>
      );
      break;
    case 'warning':
      return (
        <div className='card bg-[var(--color-warning-content)] flex-1'>
          <div className='card-body flex-1 flex flex-row items-center gap-4 justify-between'>
            <div>
              <h2 className='card-title flex-1'>{title}</h2>
              <h3 className='card-title text-sm flex-1'>{text}</h3>
            </div>
            <div className='flex flex-col items-center gap-2 text-sm'>
              <div
                className='radial-progress text-[var(--color-warning)] text-xs font-bold'
                style={
                  {
                    '--value': `${card_confidence * 100}`,
                    '--size': '3.5rem',
                    '--thickness': '0.35rem',
                  } as React.CSSProperties
                }
                aria-valuenow={card_confidence * 100}
                role='progressbar'
              >
                {Math.round(card_confidence * 100)}%
              </div>
              <span>Confidence</span>
            </div>
          </div>
        </div>
      );
      break;
    default:
      return (
        <div className='card bg-[var(--color-success-content)] flex-1'>
          <div className='card-body flex-1 flex flex-row items-center gap-4 justify-between'>
            <div>
              <h2 className='card-title flex-1'>{title}</h2>
              <h3 className='card-title text-sm flex-1'>{text}</h3>
            </div>
            <div className='flex flex-col items-center gap-2 text-sm'>
              <div
                className='radial-progress text-[var(--color-success)] text-xs font-bold'
                style={
                  {
                    '--value': `${card_confidence * 100}`,
                    '--size': '3.5rem',
                    '--thickness': '0.35rem',
                  } as React.CSSProperties
                }
                aria-valuenow={card_confidence * 100}
                role='progressbar'
              >
                {Math.round(card_confidence * 100)}%
              </div>
              <span>Confidence</span>
            </div>
          </div>
        </div>
      );
      break;
  }
}
