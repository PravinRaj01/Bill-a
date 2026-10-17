"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Download, Loader2, Share2 } from "lucide-react";
import { toast } from "sonner";

import { ReceiptImage, type ReceiptImageData } from "@/components/receipt/ReceiptImage";
import { Button } from "@/components/ui/button";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { Switch } from "@/components/ui/switch";
import type { PaymentShare } from "@/hooks/use-payment-share";
import { avatarsFor } from "@/lib/ui/avatar";
import { allOwedToMe } from "@/lib/split/settle";
import { isWebKit, renderPng, saveImage, shareImage } from "@/lib/ui/share-image";

// "Share as image": a preview of the exact picture that will be sent (it IS the generated PNG), a switch
// for adding your payment details and QR, and Share / Save. The picture is drawn from a hidden full-size
// copy of the receipt while the sheet is open; it is re-drawn when the switch changes.
export function ShareSheet({
  open,
  onOpenChange,
  data,
  payment,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  data: Omit<ReceiptImageData, "payment">;
  payment: PaymentShare;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [png, setPng] = useState<{ blob: Blob; url: string } | null>(null);
  const [failed, setFailed] = useState(false);
  // Portraits for the people on the receipt; the picture waits for them so it never goes out with blanks.
  const [avatars, setAvatars] = useState<Record<string, string> | null>(null);
  const names = data.shares.map((s) => s.name);
  const namesKey = JSON.stringify(names);
  // Safari needs the system font in the picture (see lib/ui/share-image.ts).
  const [systemFont] = useState(() => typeof navigator !== "undefined" && isWebKit(navigator.userAgent));

  // "Pay me" only makes sense when what is still owed is owed to the user.
  const paysMe = allOwedToMe(data.view, data.settlement?.me);
  const withPayment = paysMe && payment.include && payment.details ? payment.details : null;
  const image = useMemo<ReceiptImageData>(() => ({ ...data, payment: withPayment, avatars: avatars ?? undefined }), [data, withPayment, avatars]);
  // A stable key for "what the picture shows", so it is only redrawn when that really changes.
  const key = useMemo(() => JSON.stringify(image), [image]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    avatarsFor(JSON.parse(namesKey) as string[])
      .then((a) => !cancelled && setAvatars(a))
      .catch(() => !cancelled && setAvatars({})); // no portraits: initials
    return () => {
      cancelled = true;
    };
  }, [open, namesKey]);

  useEffect(() => {
    if (!open || avatars === null) return;
    let cancelled = false;
    let url: string | null = null;
    // Let the hidden copy render (and the QR svg mount) before capturing it.
    const timer = setTimeout(async () => {
      const node = hostRef.current?.firstElementChild as HTMLElement | null;
      if (!node) return;
      try {
        const blob = await renderPng(node, { systemFont });
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setFailed(false);
        setPng({ blob, url });
      } catch (e) {
        console.error("Receipt image failed", e);
        if (!cancelled) setFailed(true);
      }
    }, 80);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      if (url) URL.revokeObjectURL(url);
    };
  }, [open, key, systemFont, avatars]);

  const ready = open && png;
  const share = async () => {
    if (!png) return;
    const outcome = await shareImage(png.blob, data.title);
    if (outcome === "saved") toast("Saved to your downloads");
    if (outcome === "failed") toast.error("Couldn't share that. Try Save instead.");
  };
  const save = () => {
    if (!png) return;
    if (saveImage(png.blob, data.title) === "failed") toast.error("Couldn't save the image.");
    else toast("Saved to your downloads");
  };

  return (
    <>
      {open && (
        // Full-size copy that the picture is captured from, kept off screen. The capture is of the child, so
        // the off-screen position doesn't end up in the image.
        <div ref={hostRef} aria-hidden style={{ position: "fixed", left: -10000, top: 0, pointerEvents: "none" }}>
          <ReceiptImage data={image} systemFont={systemFont} />
        </div>
      )}

      <Drawer open={open} onOpenChange={onOpenChange}>
        <DrawerContent>
          <DrawerHeader>
            <DrawerTitle className="text-white">Share as image</DrawerTitle>
            <DrawerDescription className="text-muted-foreground">One picture with who owes what{withPayment ? " and how to pay you" : ""}.</DrawerDescription>
          </DrawerHeader>

          <div className="mx-auto w-full max-w-md space-y-4 px-4 pb-6">
            <div data-vaul-no-drag data-testid="share-preview" className="max-h-[48vh] overflow-y-auto rounded-2xl border border-white/10 bg-background">
              {ready ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={png.url} alt="Receipt preview" className="block w-full" />
              ) : failed ? (
                <p className="p-6 text-center text-sm text-coral">Couldn&apos;t make the picture. Close this and try again.</p>
              ) : (
                <div className="flex h-48 items-center justify-center text-muted-foreground">
                  <Loader2 className="size-5 animate-spin" />
                </div>
              )}
            </div>

            {paysMe && payment.details && (
              <label data-testid="image-payment-toggle" className="flex min-h-11 items-center justify-between gap-3 px-1 text-sm">
                <span className="text-muted-foreground">Include my payment details and QR</span>
                <Switch checked={payment.include} onCheckedChange={payment.setInclude} aria-label="Include my payment details and QR" />
              </label>
            )}
            {paysMe && !payment.details && payment.signedIn && (
              <p className="px-1 text-sm text-muted-foreground">
                Want friends to be able to pay from this picture?{" "}
                <Link href="/dashboard/account" className="font-semibold text-white underline underline-offset-4">
                  Add payment details
                </Link>
              </p>
            )}

            <div className="flex gap-3">
              <Button
                type="button"
                data-testid="image-share"
                disabled={!ready}
                onClick={share}
                className="h-12 flex-1 rounded-xl bg-coral text-base font-bold text-coral-foreground hover:bg-coral/90"
              >
                <Share2 className="mr-2 size-4" /> Share
              </Button>
              <Button
                type="button"
                variant="outline"
                data-testid="image-save"
                disabled={!ready}
                onClick={save}
                className="h-12 flex-1 rounded-xl border-white/10 text-base font-semibold text-zinc-200"
              >
                <Download className="mr-2 size-4" /> Save
              </Button>
            </div>
          </div>
        </DrawerContent>
      </Drawer>
    </>
  );
}
