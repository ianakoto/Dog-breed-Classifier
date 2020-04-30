import { TestBed } from '@angular/core/testing';

import { BackendconnectionService } from './backendconnection.service';

describe('BackendconnectionService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: BackendconnectionService = TestBed.get(BackendconnectionService);
    expect(service).toBeTruthy();
  });
});
