'use client';

import { useTheme } from 'next-themes';
import { Toaster as Sonner, ToasterProps } from 'sonner';

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = 'system' } = useTheme();

  return (
    <Sonner
      toastOptions={{
        // Allows complete control — no default styles
        // unstyled: true,
        // Add Tailwind (or custom CSS) to shape your toast
        classNames: {
          toast: '!bg-white/30 !backdrop-filter !backdrop-blur-sm !border-gray-200 !border !rounded-3xl !shadow-lg !px-5 !w-auto justify-self-center',
          title: 'text-lg',
          description: 'text-sm text-gray-600',
          actionButton: 'ml-2 px-3 py-1 rounded-full bg-blue-500 text-white',
          closeButton: 'ml-2 text-gray-400 hover:text-gray-600',
        },
      }}
      {...props}
    />
  );
};

export { Toaster };
