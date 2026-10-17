import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedView } from '@/components/themed-view';
import { Text } from '@/components/ui/text';
import { ThemedText } from '@/components/themed-text';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { router } from 'expo-router';
import { styles } from './stylesheet';
import { useState } from 'react';

export default function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  
  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <Button variant='ghost' size='icon' className='self-start' onPress={() => router.back()}>
          <Text className='text-3xl'>{'<'}</Text>
        </Button>

        <ThemedText className='mb-32' type='title' style={styles.title}>
          Create an account
        </ThemedText>

        <Input className='h-16 text-lg'
          keyboardType='default' placeholder='Username'
          autoCapitalize='none' autoCorrect={false}
          onChangeText={setUsername}>
        </Input>
        <Input className='h-16 text-lg'
          secureTextEntry textContentType='newPassword' autoComplete='new-password'
          autoCapitalize='none' autoCorrect={false}
          placeholder='Password'
          onChangeText={setPassword}>
        </Input>

        <Button className='h-fit'>
            <Text className='w-full text-xl text-center font-bold'>Register</Text>
        </Button>
      </SafeAreaView>
    </ThemedView>
  );
}