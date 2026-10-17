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
            variant="destructive"
            size="lg"
            data-testid="confirm-yes"
            onClick={async () => {
              onOpenChange(false);
              await onConfirm();
            }}
          >
            {confirmLabel}
          </Button>
          <Button type="button" variant="glass" size="lg" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  );
}
