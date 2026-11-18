import * as Device from 'expo-device';
import { Platform, View } from 'react-native';
import { styles } from './stylesheet';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AnimatedIcon } from '@/components/animated-icon';
import { HintRow } from '@/components/hint-row';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
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
          <View className='mt-16'>
            <AnimatedIcon />
          </View>

          <ThemedText type='title' className='text-ink mb-safe-or-48' style={styles.title}>
            Start your Hindi learning journey
          </ThemedText>

          <ThemedView style={styles.heroSection}>
            <Button className='h-fit py-4 rounded-2xl bg-primary' onPress={handleRegister}>
              <Text className='w-full font-dm-semibold text-xl text-center text-white'>Register</Text>
            </Button>
            <Button variant='secondary' className='border-ink border h-fit py-4 rounded-2xl' onPress={handleLogin}>
              <Text className='w-full font-dm-semibold text-xl text-center font-bold text-ink'>Login</Text>
            </Button>

            <Button variant='ghost' onPress={handleSkip}>
              <Text className='text-ink'>Skip to main screen</Text>
            </Button>
          </ThemedView>
        </ThemedView>
      </SafeAreaView>
    </ThemedView>
  );
}
