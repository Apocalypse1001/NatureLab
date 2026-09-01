/** WebSocket client: JSON control/state messages + versioned bulk frames. */
import type { EngineInfo, ObjectData, SimStateMessage, WorldData } from '../world/types';

export type BackendStatus = 'connecting' | 'connected' | 'disconnected';

interface Handlers {
  onStatus?: (status: BackendStatus) => void;
  onHello?: (engine: EngineInfo, simStatus: string) => void;
  onWorld?: (world: WorldData, simStatus: string) => void;
  onSimState?: (state: SimStateMessage) => void;
  onSaved?: (name: string, path: string) => void;
  onError?: (message: string) => void;
  onObjectAdded?: (obj: ObjectData) => void;
  onTerrainPatch?: (heights: number[], checksum: string) => void;
}

const MAGIC = 0x4c4e; // bytes 'N','L' read as little-endian u16

export class BackendClient {
  status: BackendStatus = 'disconnected';
  engineInfo: EngineInfo | null = null;

  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;

  constructor(private url: string, private handlers: Handlers) {}

  connect(): void {
    this.status = 'connecting';
    this.handlers.onStatus?.('connecting');
    const ws = new WebSocket(this.url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      this.status = 'connected';
      this.handlers.onStatus?.('connected');
      this.send({ op: 'request_world' });
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data === 'string') {
        try {
          const parsed: unknown = JSON.parse(ev.data);
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            this.handleText(parsed as Record<string, unknown>);
          } else {
            this.handlers.onError?.('backend message must be an object');
          }
        } catch {
          this.handlers.onError?.('backend sent invalid JSON');
        }
        return;
      }
      this.handleBinary(ev.data as ArrayBuffer);
    };

    ws.onclose = () => {
      this.status = 'disconnected';
      this.handlers.onStatus?.('disconnected');
      this.scheduleReconnect();
    };
    ws.onerror = () => ws.close();
    this.ws = ws;
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
  }

  private handleText(msg: Record<string, unknown>): void {
    switch (msg.type) {
      case 'hello':
        this.engineInfo = msg.engine as EngineInfo;
        this.handlers.onHello?.(this.engineInfo, msg.status as string);
        break;
      case 'world':
        this.handlers.onWorld?.(msg.world as WorldData, msg.status as string);
        break;
      case 'sim_state':
        this.handlers.onSimState?.(msg as unknown as SimStateMessage);
        break;
      case 'saved':
        this.handlers.onSaved?.(msg.name as string, msg.path as string);
        break;
      case 'terrain_patch':
        if (Array.isArray(msg.heights) && typeof msg.checksum === 'string') {
          this.handlers.onTerrainPatch?.(msg.heights as number[], msg.checksum);
        }
        break;
      case 'ack':
        if (msg.op === 'object_add' && msg.object) {
          this.handlers.onObjectAdded?.(msg.object as ObjectData);
        }
        break;
      case 'error':
        this.handlers.onError?.(String(msg.error));
        break;
      default:
        break; // ack messages need no UI reaction
    }
  }

  private handleBinary(buffer: ArrayBuffer): void {
    if (buffer.byteLength < 16) return;
    const view = new DataView(buffer);
    // header: magic(u16), version(u8), kind(u8), count(u32), simTimeMs(u64)
    if (buffer.byteLength < 16 || view.getUint16(0, true) !== MAGIC) return;
    if (view.getUint8(2) !== 2) return;
    const kind = view.getUint8(3);
    const count = view.getUint32(4, true);
    const components = [3, 1, 3, 10, 3, 1][kind];
    if (!components || buffer.byteLength !== 16 + count * components * 4) return;
    const floats = new Float32Array(buffer, 16, count * components);
    if (kind === 0) this.particleHandler?.(floats, count);
    if (kind === 1) this.waterHeightHandler?.(floats, count,
      Number(view.getBigUint64(8, true)) / 1000);
    if (kind === 2) this.velocityFieldHandler?.(floats, count);
  }

  particleHandler: ((positions: Float32Array, count: number) => void) | null = null;
  waterHeightHandler: ((heights: Float32Array, count: number,
                         simTime: number) => void) | null = null;
  velocityFieldHandler: ((velocities: Float32Array, count: number) => void) | null = null;

  send(op: Record<string, unknown>): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(op));
      return true;
    }
    return false;
  }
}
