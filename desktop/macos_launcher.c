/* Native Finder entry point for the source checkout. Rebuild for each Mac CPU. */
#include <mach-o/dyld.h>
#include <limits.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(void) {
    char raw[PATH_MAX], resolved[PATH_MAX], command[PATH_MAX], script[PATH_MAX];
    uint32_t size = sizeof(raw);
    if (_NSGetExecutablePath(raw, &size) || !realpath(raw, resolved)) return 1;
    char *directory = dirname(resolved);
#ifdef KAMIAK_PORTABLE
    if (snprintf(command,sizeof(command),"%s/../Resources/python/bin/python3",directory)>=(int)sizeof(command)) return 1;
    if (snprintf(script,sizeof(script),"%s/../Resources/desktop/kamiak_web.py",directory)>=(int)sizeof(script)) return 1;
    execl(command,command,"-I",script,(char *)NULL);
#else
    if (snprintf(command,sizeof(command),"%s/../../../Kamiak DI.command",directory)>=(int)sizeof(command)) return 1;
    execl("/bin/bash","bash",command,(char *)NULL);
#endif
    perror("Could not launch Kamiak DI");
    return 1;
}
