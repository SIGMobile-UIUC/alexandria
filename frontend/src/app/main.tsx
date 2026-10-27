import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedView } from '@/components/themed-view';
import { Text } from '@/components/ui/text';
import { ThemedText } from '@/components/themed-text';
import { Button } from '@/components/ui/button';
import { Input } from "@/components/ui/input"
import { router } from 'expo-router';
import { styles } from './stylesheet';
import { useState } from 'react';
import { Keyboard, TouchableWithoutFeedback } from 'react-native';

export default function Main() {
  return (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
    <ThemedView style={styles.container} className='items-center'>
      <SafeAreaView style={styles.safeArea}>
        <Text className='text-3xl font-bold'>Hindi text</Text>
        <Input keyboardType='default' returnKeyType='done' className='h-20 text-3xl' placeholder='English translation'></Input>
      </SafeAreaView>
    </ThemedView>
    </TouchableWithoutFeedback>
  )
}