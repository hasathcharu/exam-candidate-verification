export default function Emphasis({
  children,
  type,
}: {
  children: React.ReactNode;
  type: 'green' | 'red' | 'ai';
}) {
  switch (type) {
    case 'green':
      return (
        <span className='successgradient text py-1 px-2 rounded-sm inline-flex items-center gap-2 justify-center'>{children}</span>
      );
    case 'red':
      return (
        <span className='errorgradient text-white rounded-sm px-2 py-1 inline-flex items-center gap-2 justify-center'>{children}</span>
      );
    case 'ai':
      return (
        <span className='aigradient py-1 px-2 rounded-sm inline-flex items-center gap-2 justify-center'>{children}</span>
      );
  }
}
