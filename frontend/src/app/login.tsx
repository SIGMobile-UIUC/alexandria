import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedView } from '@/components/themed-view';
import { Text } from '@/components/ui/text';
import { ThemedText } from '@/components/themed-text';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { router } from 'expo-router';
import { Loader2 } from 'lucide-react-native';
import { Icon } from '@/components/ui/icon';
import { post } from '@/lib/utils';
import { setToken } from '@/lib/session';
import { styles } from './stylesheet';
import { DismissKeyboard } from '@/components/dismiss-keyboard';
import { useState } from 'react';
import { View } from 'react-native';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleLogin() {
    setLoading(true);
    setError(null);
    try {
      const res = await post<{ token: string }>('login', { username, password }, { invalidMessage: 'Enter your username and password' });
      setToken(res.token);
      router.replace('/main');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong');
      setLoading(false);
    }
  }
  
  return (
    <DismissKeyboard>
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <Button variant='ghost' size='icon' className='self-start' onPress={() => router.back()}>
          <Text className='text-3xl'>{'<'}</Text>
        </Button>

        <ThemedText className='text-ink mb-32' type='title' style={styles.title}>
          Sign in to your account
        </ThemedText>

        <Input returnKeyType='done' className='h-16 text-lg bg-secondary dark:bg-secondary border-primary text-ink placeholder:text-ink/50'
          keyboardType='default' placeholder='Username'
          autoCapitalize='none' autoCorrect={false}
          onChangeText={setUsername}>
        </Input>
        <Input returnKeyType='done' className='h-16 text-lg bg-secondary dark:bg-secondary border-primary text-ink placeholder:text-ink/50'
          secureTextEntry textContentType='password' autoComplete='current-password'
          autoCapitalize='none' autoCorrect={false}
          placeholder='Password'
          onChangeText={setPassword}>
        </Input>

        <Button className='h-fit py-4 rounded-2xl' disabled={loading} onPress={handleLogin}>
          {loading && (
            <View className="animate-spin">
              <Icon as={Loader2} className="text-primary-foreground" />
            </View>
          )}
          <Text className='font-dm-semibold text-xl text-center text-white'>Sign in</Text>
        </Button>

        {error && <Text className='text-red-600 text-center'>{error}</Text>}
      </SafeAreaView>
    </ThemedView>
    </DismissKeyboard>
  );
}