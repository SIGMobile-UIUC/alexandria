import type { ReactElement } from 'react';
import { Keyboard, TouchableWithoutFeedback } from 'react-native';

/** Wraps a single screen-level child so tapping outside any input dismisses the keyboard. */
export function DismissKeyboard({ children }: { children: ReactElement }) {
  return (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
      {children}
    </TouchableWithoutFeedback>
  );
}
