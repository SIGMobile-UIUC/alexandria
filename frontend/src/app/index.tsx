import * as Device from 'expo-device';
import { Platform } from 'react-native';
import { styles } from './stylesheet';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AnimatedIcon } from '@/components/animated-icon';
import { HintRow } from '@/components/hint-row';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { WebBadge } from '@/components/web-badge';
import { Button } from '@/components/ui/button';
import { Text } from '@/components/ui/text';
import { router } from 'expo-router';

export default function HomeScreen() {

  const handleRegister = () => router.push('/register');
  const handleLogin = () => router.push('/login');
  const handleSkip = () => router.push('/main');

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ThemedView style={styles.heroSection}>
          <AnimatedIcon />
          <ThemedText type='title' style={styles.title}>
            Start learning a language today.
          </ThemedText>

          <Button className='h-fit' onPress={handleRegister}>
            <Text className='w-full text-xl text-center font-bold'>Create an account</Text>
          </Button>
          <Button variant='secondary' className='h-fit' onPress={handleLogin}>
            <Text className='w-full text-xl text-center font-bold'>Sign in</Text>
          </Button>

          <Button onPress={handleSkip}>
            <Text>Skip to main screen</Text>
          </Button>
        </ThemedView>


        {Platform.OS === 'web' && <WebBadge />}
      </SafeAreaView>
    </ThemedView>
  );
}
