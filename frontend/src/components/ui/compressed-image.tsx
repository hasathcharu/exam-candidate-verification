import imageCompression from 'browser-image-compression';
import React, { useEffect, useState } from 'react';

export default function CompressedImage({ img }: { img: File }) {
  const [compressedImage, setCompressedImage] = useState<File>();

  useEffect(() => {
    imageCompression(img, {
      maxSizeMB: 1,
      maxWidthOrHeight: 1024,
      useWebWorker: true,
    }).then((compressedBlob) => {
      setCompressedImage(
        new File([compressedBlob], img.name, {
          type: compressedBlob.type,
          lastModified: compressedBlob.lastModified,
        })
      );
    });
  }, [img]);

  if (!compressedImage) return null;

  return (
    <img
      src={URL.createObjectURL(compressedImage)}
      alt={compressedImage.name}
      className="size-full object-cover"
    />
  );
}
