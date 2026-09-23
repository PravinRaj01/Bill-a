import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Bill.a - The Splitter",
  description: "AI-Powered Bill Splitter",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Bill.a",
  },
  icons: {
  icon: '/icon.png',
  shortcut: '/favicon.ico',
  apple: [
    { url: '/apple-icon.png', sizes: '180x180', type: 'image/png' },
  ],
  },
  formatDetection: {
    telephone: false, // Prevents phone numbers from turning blue/clickable
  },
};

export const viewport: Viewport = {
  themeColor: "#09090b",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false, // Prevents accidental zooming on input focus
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" style={{ colorScheme: 'dark' }}>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#09090b] text-slate-50`}>
        {children}
      </body>
    </html>
  );
}