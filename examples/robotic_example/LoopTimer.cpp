#include "LoopTimer.h"

#ifndef USE_CHRONO
// Helper timespec functions
static inline timespec operator-(const timespec& a, const timespec& b) {
	timespec dt;
	if (a.tv_nsec - b.tv_nsec < 0) {
		dt.tv_sec  = a.tv_sec  - b.tv_sec  - 1;
		dt.tv_nsec = a.tv_nsec - b.tv_nsec + 1e9;
	} else {
		dt.tv_sec  = a.tv_sec  - b.tv_sec;
		dt.tv_nsec = a.tv_nsec - b.tv_nsec;
	}
	return dt;
}

static inline bool operator<(const timespec& lhs, const timespec& rhs) {
	if (lhs.tv_sec == rhs.tv_sec)
		return lhs.tv_nsec < rhs.tv_nsec;
	return lhs.tv_sec < rhs.tv_sec;
}

static inline timespec& operator+=(timespec& t, unsigned int nsecs) {
	while (nsecs >= 1e9) {
		t.tv_sec++;
		nsecs -= 1e9;
	}
	t.tv_nsec += nsecs;
	return t;
}

static inline double timespec_to_double(const timespec& t) {
	return t.tv_sec + 1e-9 * static_cast<double>(t.tv_nsec);
}
#endif  // USE_CHRONO

void LoopTimer::setLoopFrequency (double frequency) {
#ifdef USE_CHRONO
	ns_update_interval_ = std::chrono::nanoseconds(static_cast<unsigned int>(1e9 / frequency));
#else  // USE_CHRONO
	ns_update_interval_ = 1e9 / frequency;
#endif  // USE_CHRONO
}

void LoopTimer::initializeTimer(unsigned int initial_wait_nanoseconds) {
#ifdef USE_CHRONO
	auto ns_initial_wait = std::chrono::nanoseconds(initial_wait_nanoseconds);
	t_next_ = std::chrono::high_resolution_clock::now() + ns_initial_wait;
	t_start_ = t_next_;
	t_loop_ = t_start_ - t_start_;
#else  // USE_CHRONO
	// initialize time
	getCurrentTime(t_next_);

	// calculate next shot. carry over nanoseconds into seconds.
	t_next_ += initial_wait_nanoseconds;
	t_start_ = t_next_;
	// TODO os x
	// http://stackoverflow.com/questions/11338899/are-there-any-well-behaved-posix-interval-timers
#endif  // USE_CHRONO
}

bool LoopTimer::waitForNextLoop() {
#ifdef USE_CHRONO
	bool slept = false;
	t_curr_ = std::chrono::high_resolution_clock::now();
	if (t_curr_ < t_next_) {
		std::this_thread::sleep_for(t_next_ - t_curr_);
		slept = true;
	}
	t_curr_ = std::chrono::high_resolution_clock::now();
	t_loop_ = t_curr_ - t_start_;
	t_next_ += ns_update_interval_;
	update_counter_++;
	return slept;
#else  // USE_CHRONO
	// grab the time
	getCurrentTime(t_curr_);

	// wait until next shot if necessary (this check is redundant for linux)
	bool slept = false;
	if (t_curr_ < t_next_) {
		nanoSleepUntil(t_next_, t_curr_);
		slept = true;
	}

	// calculate dt
	t_loop_ = t_curr_ - t_start_;

	// calculate next shot
	t_next_ += ns_update_interval_;

	// increment loop counter
	++update_counter_;

	return slept;
#endif  // USE_CHRONO
}

unsigned long long LoopTimer::elapsedCycles() {
	return update_counter_;
}

double LoopTimer::loopTime() {
#ifdef USE_CHRONO
	return std::chrono::duration<double>(t_loop_).count();
#else  // USE_CHRONO
	return timespec_to_double(t_loop_);
#endif  // USE_CHRONO
}

double LoopTimer::elapsedTime() {
#ifdef USE_CHRONO
	return std::chrono::duration<double>(t_loop_).count();
#else  // USE_CHRONO
	struct timespec t;
	elapsedTime(t);
	return timespec_to_double(t);
#endif  // USE_CHRONO
}

double LoopTimer::elapsedSimTime() {
#ifdef USE_CHRONO
	return update_counter_ * std::chrono::duration<double>(ns_update_interval_).count();
#else  // USE_CHRONO
	return update_counter_ * (1e-9 * ns_update_interval_);
#endif  // USE_CHRONO
}

#ifndef USE_CHRONO
void LoopTimer::loopTime(timespec& t) {
	t = t_loop_;
}

void LoopTimer::elapsedTime(timespec& t) {
	struct timespec t_now;
	getCurrentTime(t_now);
	t = t_now - t_start_;
}
#endif  // USE_CHRONO


void LoopTimer::run(void (*userCallback)(void)) {
#ifdef USE_CHRONO
	initializeTimer(ns_update_interval_.count());
#else  // USE_CHRONO
	initializeTimer(ns_update_interval_);
#endif  // USE_CHRONO

	running_ = true;
	while (running_) {
		waitForNextLoop();
		userCallback();
	}
}

void LoopTimer::stop() {
	running_ = false;
}

// static void LoopTimer::setThreadHighPriority(){
//     pid_t pid = getpid();
//     int priority_status = setpriority(PRIO_PROCESS, pid, -19);
//     if (priority_status){
//         printWarning("setThreadHighPriority. Failed to set priority.");
//     }
// }

// static void LoopTimer::setThreadRealTime(const int MAX_SAFE_STACK = 8*1024) {
//     // Declare ourself as a real time task, priority 49.
//     // PRREMPT_RT uses priority 50
//     // for kernel tasklets and interrupt handler by default
//     struct sched_param param;
//     param.sched_priority = 49;
//    if(sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
//         perror("sched_setscheduler failed");
//         exit(-1);
//     }

//     // Lock memory
//     if(mlockall(MCL_CURRENT|MCL_FUTURE) == -1) {
//         perror("mlockall failed");
//         exit(-2);
//     }

//     // Pre-fault our stack
//     //int MAX_SAFE_STACK = 8*1024;
//     unsigned char dummy[MAX_SAFE_STACK];
//     memset(dummy, 0, MAX_SAFE_STACK);
// }

#ifndef USE_CHRONO
inline void LoopTimer::getCurrentTime(timespec &t_ret) {
#ifdef __APPLE__
	static double ratio = 0.0;
	if (!ratio) {
		mach_timebase_info_data_t info;
		if (mach_timebase_info(&info) == KERN_SUCCESS) {
			ratio = info.numer * info.denom;
		}
	}
	uint64_t t_nsecs = mach_absolute_time() * ratio;
	t_ret.tv_sec = 0;
	while (t_nsecs >= 1000000000){
		t_nsecs -= 1000000000;
		t_ret.tv_sec++;
	}
	t_ret.tv_nsec = static_cast<long>(t_nsecs);
#else  // __APPLE__
	clock_gettime(CLOCK_MONOTONIC, &t_ret);
#endif  // __APPLE__
}

inline void LoopTimer::nanoSleepUntil(const timespec &t_next, const timespec &t_now) {
#ifdef __APPLE__
	timespec t_sleep = t_next - t_now;
	nanosleep(&t_sleep, NULL);
#else  // __APPLE__
	clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t_next_, NULL);
#endif  // __APPLE__
}
#endif  // USE_CHRONO
