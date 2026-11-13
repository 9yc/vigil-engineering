// Sample two integer-valued files on an absolute monotonic schedule.
// Values are recorded as raw integers; units and scaling belong to the caller.
#include <cerrno>
#include <cctype>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

namespace {
volatile sig_atomic_t stopRequested = 0;
constexpr long intervalNs = 100000000;

void stop(int) { stopRequested = 1; }

[[noreturn]] void fail(const char* message) {
  fprintf(stderr, "sample_nodes: %s: %s\n", message, strerror(errno));
  exit(1);
}

long parseInteger(const char* text, long minimum, long maximum,
                  const char* label) {
  errno = 0;
  char* end = nullptr;
  long value = strtol(text, &end, 10);
  if (errno || end == text) {
    errno = EINVAL;
    fail(label);
  }
  while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;
  if (*end || value < minimum || value > maximum) {
    errno = EINVAL;
    fail(label);
  }
  return value;
}

uint64_t readClock(clockid_t id) {
  struct timespec ts;
  if (clock_gettime(id, &ts)) fail("read clock");
  return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

int readNode(int fd) {
  if (lseek(fd, 0, SEEK_SET) < 0) fail("seek input");
  char buffer[64];
  ssize_t count;
  do {
    count = read(fd, buffer, sizeof(buffer) - 1);
  } while (count < 0 && errno == EINTR);
  if (count < 0) fail("read input");
  if (count == 0 || count == ssize_t(sizeof(buffer) - 1)) {
    errno = EINVAL;
    fail("empty or oversized input");
  }
  buffer[count] = '\0';
  if (memchr(buffer, '\0', static_cast<size_t>(count))) {
    errno = EINVAL;
    fail("embedded null in input");
  }
  return static_cast<int>(
      parseInteger(buffer, INT_MIN, INT_MAX, "expected a signed 32-bit integer"));
}

void advance(struct timespec& ts) {
  ts.tv_nsec += intervalNs;
  if (ts.tv_nsec >= 1000000000) {
    ++ts.tv_sec;
    ts.tv_nsec -= 1000000000;
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    fprintf(stderr, "usage: %s output.csv node_a node_b [sample_count]\n"
                    "sample_count: positive integer, or 0 for continuous sampling\n",
            argv[0]);
    return 2;
  }
  const long limit = argc == 5
      ? parseInteger(argv[4], 0, LONG_MAX, "invalid sample_count") : 0;
  const int fdA = open(argv[2], O_RDONLY | O_CLOEXEC);
  if (fdA < 0) fail("open node_a");
  const int fdB = open(argv[3], O_RDONLY | O_CLOEXEC);
  if (fdB < 0) fail("open node_b");
  // Validate inputs before creating an output file.
  readNode(fdA);
  readNode(fdB);
  const int outputFd =
      open(argv[1], O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (outputFd < 0) fail("create output (must not already exist)");
  FILE* output = fdopen(outputFd, "w");
  if (!output) fail("open output stream");
  if (fprintf(output, "realtime_ns,monotonic_ns,value_a_raw,value_b_raw\n") < 0)
    fail("write header");
  struct sigaction action = {};
  action.sa_handler = stop;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTERM, &action, nullptr) ||
      sigaction(SIGINT, &action, nullptr))
    fail("install signal handler");
  struct timespec next;
  if (clock_gettime(CLOCK_MONOTONIC, &next)) fail("initial clock");
  long samples = 0;
  while (!stopRequested) {
    const uint64_t mono = readClock(CLOCK_MONOTONIC);
    const uint64_t real = readClock(CLOCK_REALTIME);
    const int a = readNode(fdA);
    const int b = readNode(fdB);
    if (fprintf(output, "%llu,%llu,%d,%d\n",
                static_cast<unsigned long long>(real),
                static_cast<unsigned long long>(mono), a, b) < 0 ||
        fflush(output))
      fail("write sample");
    if (limit && ++samples >= limit) break;
    advance(next);
    const uint64_t now = readClock(CLOCK_MONOTONIC);
    while (uint64_t(next.tv_sec) * 1000000000ull + next.tv_nsec <= now)
      advance(next);
    while (!stopRequested) {
      const int rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
      if (rc == 0) break;
      if (rc != EINTR) {
        errno = rc;
        fail("sleep");
      }
    }
  }
  if (fclose(output) || close(fdA) || close(fdB)) fail("close files");
  return 0;
}
