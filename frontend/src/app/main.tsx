import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedView } from '@/components/themed-view';
import { Text } from '@/components/ui/text';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { Icon } from '@/components/ui/icon';
import { DismissKeyboard } from '@/components/dismiss-keyboard';
import { router } from 'expo-router';
import { styles } from './stylesheet';
import { useEffect, useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import Animated, { FadeIn, FadeOut, SlideInDown, SlideOutDown } from 'react-native-reanimated';
import { Check, Loader2, X } from 'lucide-react-native';
import { get, post } from '@/lib/utils';
import { getToken } from '@/lib/session';

const TOTAL_QUESTIONS = 10;

type Result = 'correct' | 'almost' | 'incorrect';

const RESULTS: Record<Result, { title: string; xp: number; sheet: string; accent: string; text: string; tint: string }> = {
  correct: { title: 'Correct!', xp: 10, sheet: '#E8F3EA', accent: '#529C5E', text: '#2F6B3A', tint: 'rgba(82, 156, 94, 0.10)' },
  almost: { title: 'Almost correct', xp: 5, sheet: '#FCF0CC', accent: '#DFA53C', text: '#8A5A16', tint: 'rgba(223, 165, 60, 0.12)' },
  incorrect: { title: 'Incorrect', xp: 0, sheet: '#F8E6E3', accent: '#C9524A', text: '#A33A2E', tint: 'rgba(201, 82, 74, 0.10)' },
};

function classify(score: number): Result {
  if (score >= 90) return 'correct';
  if (score >= 60) return 'almost';
  return 'incorrect';
}

const normalize = (word: string) => word.toLowerCase().replace(/[^\p{L}\p{N}']/gu, '');

function CorrectAnswer({ answer, userAnswer, highlight }: { answer: string; userAnswer: string; highlight?: string }) {
  const userWords = new Set(userAnswer.split(/\s+/).map(normalize));
  return (
    <Text className='font-dm-semibold text-xl text-ink'>
      {answer.split(/(\s+)/).map((part, i) => {
        const missed = highlight && part.trim() && !userWords.has(normalize(part));
        return missed ? (
          <Text key={i} className='font-dm-semibold text-xl underline' style={{ color: highlight }}>
            {part}
          </Text>
        ) : (
          part
        );
      })}
    </Text>
  );
}

export default function Main() {
  const [question, setQuestion] = useState(1);
  const [sentence, setSentence] = useState<string>();
  const [correctAnswer, setCorrectAnswer] = useState('');
  const [answer, setAnswer] = useState('');
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [score, setScore] = useState(0);
  const [xpEarned, setXpEarned] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);

  async function loadSentence() {
    setSentence(undefined);
    setError(null);
    try {
      const response = await get<{ hindi: string; english: string }[]>('sentence', new URLSearchParams({ level: '5' }));
      setSentence(response[0]?.hindi);
      setCorrectAnswer(response[0]?.english ?? '');
    } catch {
      setError('Could not load a sentence');
    }
  }

  useEffect(() => {
    loadSentence();
  }, []);

  async function handleCheck() {
    if (!sentence || checking) return;
    setChecking(true);
    setError(null);
    try {
      const response = await get<{ score: number }>('check', new URLSearchParams({
        source: sentence,
        answer: answer.trim() || 'None',
        source_lang: 'hi',
      }));
      const percent = Math.round(response.score * 100);
      const outcome = classify(percent);
      setScore(percent);
      setResult(outcome);

      const xp = RESULTS[outcome].xp;
      if (xp > 0) {
        setXpEarned((total) => total + xp);
        const token = getToken();
        if (token) {
          post('xp', { amount: xp, reason: 'translation' }, { token }).catch((e) => console.warn(e));
        }
      }
    } catch {
      setError('Could not check your answer');
    } finally {
      setChecking(false);
    }
  }

  function handleContinue() {
    setResult(null);
    setAnswer('');
    if (question >= TOTAL_QUESTIONS) {
      setFinished(true);
      return;
    }
    setQuestion((q) => q + 1);
    loadSentence();
  }

  function handleRestart() {
    setQuestion(1);
    setXpEarned(0);
    setFinished(false);
    loadSentence();
  }

  const answered = question - 1 + (result ? 1 : 0);
  const progress = finished ? 100 : (answered / TOTAL_QUESTIONS) * 100;
  const style = result ? RESULTS[result] : null;

  if (finished) {
    return (
      <ThemedView style={styles.container}>
        <SafeAreaView style={styles.safeArea} className='justify-center'>
          <Text className='font-dm-semibold text-4xl text-ink text-center'>Lesson complete!</Text>
          <Text className='text-xl text-ink/70 text-center'>You earned {xpEarned} XP</Text>
          {!getToken() && (
            <Text className='text-base text-ink/60 text-center'>Sign in to save your XP</Text>
          )}
          <Button className='h-fit py-4 rounded-2xl w-full mt-6' onPress={handleRestart}>
            <Text className='font-dm-semibold text-xl text-center text-white'>Practice again</Text>
          </Button>
          <Button variant='ghost' onPress={() => router.replace('/')}>
            <Text className='text-ink'>Back to home</Text>
          </Button>
        </SafeAreaView>
      </ThemedView>
    );
  }

  return (
    <DismissKeyboard>
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <View className='w-full flex-row items-center gap-4 mb-4'>
          <Pressable onPress={() => router.replace('/')} hitSlop={12}>
            <Icon as={X} size={24} className='text-ink/70' />
          </Pressable>
          <Progress value={progress} className='flex-1 h-2.5 bg-ink/15' indicatorClassName='bg-[#B07A2A]' />
        </View>

        <Text className='w-full text-sm uppercase tracking-wider text-ink/70'>Translate this sentence</Text>
        <View className='w-full rounded-3xl border border-ink/10 bg-white px-6 py-7 shadow-sm shadow-black/10'>
          <Text className='text-3xl leading-[52px] font-bold text-ink'>{sentence ?? (error ? '' : 'Loading...')}</Text>
        </View>

        <Text className='w-full text-base text-ink/80 mt-2'>Your answer in English</Text>
        <Textarea
          value={answer}
          onChangeText={setAnswer}
          editable={!result && !checking}
          keyboardType='default'
          returnKeyType='done'
          className='h-[18vh] text-xl rounded-2xl border-2 opacity-100 bg-secondary dark:bg-secondary border-primary text-ink placeholder:text-ink/50'
          style={style ? { borderColor: style.accent, backgroundColor: style.tint } : undefined}
          placeholder='Type your translation'
        />

        {error && <Text className='text-red-600 text-center'>{error}</Text>}

        {!result && (
          <Button
            className='h-fit py-4 rounded-2xl w-full mt-auto'
            disabled={!sentence || checking || !answer.trim()}
            onPress={handleCheck}>
            {checking && (
              <View className='animate-spin'>
                <Icon as={Loader2} className='text-primary-foreground' />
              </View>
            )}
            <Text className='font-dm-semibold text-xl text-center text-white'>Check</Text>
          </Button>
        )}
      </SafeAreaView>

      {result && style && (
        <>
          <Animated.View
            entering={FadeIn.duration(200)}
            exiting={FadeOut.duration(200)}
            pointerEvents='none'
            style={[StyleSheet.absoluteFill, { backgroundColor: 'rgba(0, 0, 0, 0.3)' }]}
          />
          <Animated.View entering={SlideInDown.duration(260)} exiting={SlideOutDown.duration(220)} style={{ position: 'absolute', left: 0, right: 0, bottom: 0 }}>
          <View
            className='rounded-t-3xl px-6 pt-7 pb-12 gap-4'
            style={{ backgroundColor: style.sheet }}>
            <View className='flex-row items-center gap-4'>
              <View className='h-12 w-12 rounded-full items-center justify-center' style={{ backgroundColor: style.accent }}>
                {result === 'correct' && <Icon as={Check} size={26} className='text-white' />}
                {result === 'almost' && <Text className='font-dm-semibold text-2xl text-white'>!</Text>}
                {result === 'incorrect' && <Icon as={X} size={26} className='text-white' />}
              </View>
              <Text className='font-dm-semibold text-3xl flex-1' style={{ color: style.text }}>
                {style.title}
              </Text>
              {style.xp > 0 && (
                <Text className='font-dm-semibold text-lg' style={{ color: style.text }}>
                  +{style.xp} XP
                </Text>
              )}
            </View>

            <Text className='text-lg text-ink'>Our model gave you a similarity score of {score}%</Text>

            {result !== 'correct' && (
              <View className='rounded-2xl bg-white px-4 py-3 gap-1'>
                <Text className='text-xs uppercase tracking-wider text-ink/60'>Correct answer</Text>
                <CorrectAnswer
                  answer={correctAnswer}
                  userAnswer={answer}
                  highlight={result === 'almost' ? style.text : undefined}
                />
              </View>
            )}

            <Button className='h-fit py-4 rounded-2xl' style={{ backgroundColor: style.accent }} onPress={handleContinue}>
              <Text className='font-dm-semibold text-xl text-center text-white'>
                {question >= TOTAL_QUESTIONS ? 'Finish' : 'Continue'}
              </Text>
            </Button>
          </View>
          </Animated.View>
        </>
      )}
    </ThemedView>
    </DismissKeyboard>
  );
}
