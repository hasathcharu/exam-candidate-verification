'use client';
import React from 'react';
import {
    Dialog,
    DialogContent,
    DialogTrigger,
    DialogClose,
} from './image-dialog';
import { X } from 'lucide-react';

export default function Home(props: { [x: string]: any }) {
    return (
        <>
            <Dialog>
                <DialogTrigger>
                    <img
                        alt={props.alt}
                        src={props.src}
                        className={props.className}
                        style={{ cursor: 'pointer' }}
                    />
                </DialogTrigger>
                <DialogContent>
                    <div
                        className={
                            props?.portrait
                                ? 'image-viewer-portrait'
                                : 'image-viewer-landscape'
                        }
                    >
                        <img
                            alt={props.alt}
                            src={props.src}
                            className='box-shadow rounded-2xl object-cover'
                        />
                        <DialogClose className='absolute right-4 top-4 h-8 w-8 rounded-3xl bg-white p-1 cursor-pointer'>
                            <X className='h-6 w-6 stroke-2 align-middle' />
                            <span className='sr-only'>Close</span>
                        </DialogClose>
                    </div>
                    <div className='mx-auto max-w-[60ch] p-5 text-center text-sm italic'>
                        {props.alt}
                    </div>
                </DialogContent>
            </Dialog>
        </>
    );
}
