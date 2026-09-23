/**
 * Interface language: English (the default) or Russian.
 *
 * A string is its own key: `t('Discharge Q')` returns the Russian entry when
 * the interface is in Russian and the English text otherwise, so a string with
 * no translation yet still reads -- in English -- instead of showing a key.
 * `{name}` placeholders are filled from `vars` after the lookup, so a
 * translation can move them.
 *
 * The choice lives in localStorage and applying it reloads the page: the
 * simulation runs on the backend and carries on, and every control is rebuilt
 * already in the new language instead of being re-labelled one by one. The
 * default is English whatever the browser says, because the e2e tests and the
 * agent driver find controls by their English text.
 */
import { RU } from './i18n.ru';

export type Lang = 'en' | 'ru';

const STORAGE_KEY = 'naturelab.lang';

function storedLang(): Lang {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    if (value === 'en' || value === 'ru') return value;
  } catch { /* storage blocked: fall back to English */ }
  return 'en';
}

export const lang: Lang = storedLang();

/** Every key asked for, and (in Russian) every one with no entry -- for checks. */
export const i18nKeys = new Set<string>();
export const i18nMissing = new Set<string>();

// Numbers, units-only fragments and ids are not text to translate.
const NOT_TEXT = /^[\s\d.,:;|/()%+\-–×°=<>≈]*$|^[A-Z][a-z]*_\d+$/;

export function t(en: string, vars?: Record<string, string | number>): string {
  let text = en;
  if (en && !NOT_TEXT.test(en)) {
    i18nKeys.add(en);
    if (lang === 'ru') {
      const ru = RU[en];
      if (ru !== undefined) text = ru;
      else i18nMissing.add(en);
    }
  }
  if (vars) text = text.replace(/\{(\w+)\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m));
  return text;
}

/**
 * Translate the text runs of a small HTML fragment and leave its tags alone:
 * `<label>Pipe diameter <output id="x">200</output> mm</label>` translates
 * "Pipe diameter" and "mm", keeps the <output> and its value, and keeps the
 * spaces around each run.
 */
export function tHtml(html: string): string {
  return html.replace(/(^|>)([^<]+)/g, (_m, open: string, run: string) => {
    const lead = run.match(/^\s*/)![0];
    const trail = run.slice(lead.length).match(/\s*$/)![0];
    const core = run.slice(lead.length, run.length - trail.length);
    return open + lead + (core ? t(core) : '') + trail;
  });
}

// What a language switch carries over the reload: the view, not the world
// (the backend keeps that). Session storage, so a fresh visit starts clean.
const CARRY_KEY = 'naturelab.carry';

export interface Carry {
  camera: number[];
  target: number[];
  openSections: string[];
  scenario: string | null;
}

export function setLang(next: Lang, carry: Carry): void {
  if (next === lang) return;
  try {
    localStorage.setItem(STORAGE_KEY, next);
    sessionStorage.setItem(CARRY_KEY, JSON.stringify(carry));
  } catch { /* storage blocked: the switch cannot persist, reload anyway */ }
  location.reload();
}

/** The view a language switch left behind, once; null on an ordinary load. */
export function takeCarry(): Carry | null {
  try {
    const raw = sessionStorage.getItem(CARRY_KEY);
    sessionStorage.removeItem(CARRY_KEY);
    return raw ? JSON.parse(raw) as Carry : null;
  } catch {
    return null;
  }
}
