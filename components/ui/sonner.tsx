"use client";
import { Toaster as Sonner, type ToasterProps } from "sonner";

// The app is dark-only (html.dark), so the theme is fixed. Toasts sit above the mobile
// bottom nav and respect the iOS safe area; swipe-to-dismiss is built in.
function Toaster(props: ToasterProps) {
  return (
    <Sonner
      theme="dark"
      position="bottom-center"
      offset={{ bottom: "calc(5rem + env(safe-area-inset-bottom))" }}
      mobileOffset={{ bottom: "calc(5rem + env(safe-area-inset-bottom))" }}
      toastOptions={{
        classNames: {
          toast: "!bg-surface !text-foreground !border !border-white/10 !rounded-2xl !shadow-2xl",
          description: "!text-muted-foreground",
          actionButton: "!bg-coral !text-coral-foreground !font-bold",
        },
      }}
      {...props}
    />
  );
}

export { Toaster };
