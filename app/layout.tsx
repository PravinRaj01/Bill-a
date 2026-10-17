import type { Metadata, Viewport } from "next";
import { Geist_Mono, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import { SwRegister } from "@/components/sw-register";
import { MotionProvider } from "@/components/motion-provider";
import { OrbBackdrop } from "@/components/ui/orb-backdrop";
import { Toaster } from "@/components/ui/sonner";

const jakarta = Plus_Jakarta_Sans({ variable: "--font-jakarta", subsets: ["latin"], display: "swap" });
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
  themeColor: "#0e0f12",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" style={{ colorScheme: 'dark' }}>
      <body className={`${jakarta.variable} ${geistMono.variable} antialiased bg-background text-foreground`}>
        <OrbBackdrop />
        <MotionProvider>{children}</MotionProvider>
        <Toaster />
        <SwRegister />
      </body>
    </html>
  );
}