"use client";

import { Button } from "@/components/ui/button";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer";

// A bottom sheet in place of the browser's confirm(): themed, finger-sized buttons, drag to dismiss,
// and it names what is about to happen. For destructive actions.
export function ConfirmSheet({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmLabel: string;
  onConfirm: () => void | Promise<void>;
}) {
  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent>
        <DrawerHeader>
          <DrawerTitle className="text-white">{title}</DrawerTitle>
          <DrawerDescription className="text-muted-foreground">{description}</DrawerDescription>
        </DrawerHeader>
        <DrawerFooter className="mx-auto w-full max-w-md">
          <Button
            type="button"
            data-testid="confirm-yes"
            className="h-12 rounded-xl bg-red-500 text-base font-bold text-white hover:bg-red-500/90"
            onClick={async () => {
              onOpenChange(false);
              await onConfirm();
            }}
          >
            {confirmLabel}
          </Button>
          <Button type="button" variant="outline" className="h-12 rounded-xl border-white/10 text-base text-zinc-200" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  );
}
